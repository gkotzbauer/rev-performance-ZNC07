#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
01_preprocess_and_enhance.py  (Invoice-centric; no Visit_Count)
- Drops any Visit_Count/Visit Count columns (a "visit" == an invoice).
- Computes all metrics at the invoice level.
- Produces an underpayment summary (benchmark-family blocks + 85EM passthrough if present).

Outputs:
  /mnt/data/invoice_level_index_enhanced.csv
  /mnt/data/underpayment_summary.csv
  /mnt/data/preprocess_and_enhance_bundle.zip
"""

import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
import zipfile

DATA_DIR = Path(os.getenv("DATA_DIR", "/mnt/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Input auto-detect (override with SOURCE_FILE env)
SOURCE_FILE = os.getenv("SOURCE_FILE")
if not SOURCE_FILE:
    candidates = list(DATA_DIR.glob("*Rev*Perf*Second*Layer*.xlsx")) + \
                 list(DATA_DIR.glob("*Rev*Perf*.xlsx")) + \
                 list(DATA_DIR.glob("*.xlsx")) + \
                 list(DATA_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError("No input file found in /mnt/data. Set SOURCE_FILE=...")
    SOURCE_FILE = str(sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0])

OUT_INVOICE = DATA_DIR / "invoice_level_index_enhanced.csv"
OUT_UNDERPAY = DATA_DIR / "underpayment_summary.csv"
OUT_ZIP = DATA_DIR / "preprocess_and_enhance_bundle.zip"

def to_float_safe(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.integer, np.floating)): return float(x)
    s = str(x).strip().replace(",", "")
    if s == "" or s.lower() in {"na","nan","none","<na>","null"}: return np.nan
    if s.endswith("%"):
        try: return float(s[:-1]) / 100.0
        except: return np.nan
    try: return float(s)
    except: return np.nan

def read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    return pd.read_excel(p, sheet_name=0)

# --- Load & clean ------------------------------------------------------------
df = read_any(SOURCE_FILE)
df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed", na=False)].copy()

# Standardize core names
rename_map = {
    "Year of Visit Service Date": "Year",
    "ISO Week of Visit Service Date": "Week",
    "Primary Financial Class": "Payer",
    "Chart E/M Code Grouping": "Group_EM",
    "Chart E/M Code Second Layer": "Group_EM2",
    "Charge Invoice Number": "Invoice_Number",
    "Payment Amount*": "Payment_Amount",
    "Payment Amount": "Payment_Amount",
    "Charge CPT Code": "CPT_Code",
    "Zero Balance - Collection * Charges": "Zero_Balance_Collection_Star_Charges",
    "Zero Balance Collection Rate": "Zero_Balance_Collection_Rate",
    "Collection Rate*": "Collection_Rate",
    "Denial %": "Denial_Percent",
    "NRV Zero Balance*": "NRV_Zero_Balance",
    "Charge Billed Balance": "Charge_Billed_Balance",
    "Expected Amount (85% E/M)": "Expected_Amount_85_EM_invoice_level",
    "Expected Amount 85% E/M": "Expected_Amount_85_EM_invoice_level",
    "NRV Gap ($)": "NRV_Gap_Dollar",
    "NRV Gap (%)": "NRV_Gap_Percent",
    "NRV Gap Sum ($)": "NRV_Gap_Sum_Dollar",
    "% of Remaining Charges": "Remaining_Charges_Percent",
    # Explicitly DO NOT map any visit counts
}

for k,v in rename_map.items():
    if k in df.columns and v not in df.columns:
        df.rename(columns={k:v}, inplace=True)

# Normalize core identity fields
for c in ["Year","Week","Payer","Group_EM","Group_EM2","Invoice_Number"]:
    if c in df.columns:
        df[c] = df[c].ffill()

# Extract ints for Year/Week
if "Year" in df.columns:
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
if "Week" in df.columns:
    df["Week"] = df["Week"].astype(str).str.extract(r"(\d+)", expand=False)
    df["Week"] = pd.to_numeric(df["Week"], errors="coerce").astype("Int64")

# Numeric coercion for currency/percent; DO NOT create Visit_Count
currency_like = [c for c in df.columns if any(tok in c for tok in ["Amount","Balance","NRV","Dollar","$","Payment_Amount"])]
percent_like  = [c for c in df.columns if any(tok in c for tok in ["%","Rate","Percent"])]
for c in set(currency_like + percent_like):
    df[c] = df[c].apply(to_float_safe)

# Drop any Visit count columns if present
for vc in ["Visit_Count","Visit Count","Open Visit Count","Visits"]:
    if vc in df.columns:
        df.drop(columns=[vc], inplace=True)

# CPT list per invoice
if "CPT_Code" in df.columns:
    tmp = (df.assign(CPT_Code=df["CPT_Code"].astype(str).str.strip())
             .groupby("Invoice_Number", dropna=False)["CPT_Code"]
             .apply(lambda x: sorted(set([z for z in x if z and z != "nan"])))
             .reset_index())
    tmp["CPT_List_Str"] = tmp["CPT_Code"].apply(lambda L: ",".join(L))
    df = df.merge(tmp[["Invoice_Number","CPT_List_Str"]], on="Invoice_Number", how="left")
else:
    df["CPT_List_Str"] = ""

# Keys
for c in ["Payer","Group_EM","Group_EM2","CPT_List_Str"]:
    if c not in df.columns: df[c] = ""
df["Benchmark_Key"] = (
    df["Payer"].astype(str) + "|" +
    df["Group_EM"].astype(str) + "|" +
    df["Group_EM2"].astype(str) + "|" +
    df["CPT_List_Str"].astype(str)
)

# Ensure Payment_Amount exists
if "Payment_Amount" not in df.columns:
    alt = [c for c in df.columns if "Payment" in c and "Amount" in c]
    if alt:
        df["Payment_Amount"] = df[alt[0]].apply(to_float_safe)
    else:
        df["Payment_Amount"] = 0.0

# --- Ensure NRV gap columns exist (pass-through if present) -------------------
if "NRV_Gap_Dollar" not in df.columns:
    if "Expected_Amount_85_EM_invoice_level" in df.columns:
        df["NRV_Gap_Dollar"] = (df["Expected_Amount_85_EM_invoice_level"] - df["Payment_Amount"]).astype(float)
    else:
        df["NRV_Gap_Dollar"] = np.nan

if "NRV_Gap_Percent" not in df.columns:
    denom = df.get("Expected_Amount_85_EM_invoice_level", np.nan)
    df["NRV_Gap_Percent"] = np.where(pd.notna(denom) & (denom != 0),
                                     df["NRV_Gap_Dollar"]/denom, np.nan)

if "NRV_Gap_Sum_Dollar" not in df.columns:
    df["NRV_Gap_Sum_Dollar"] = df["NRV_Gap_Dollar"]

# --- Underpayment summary (invoice-level) ------------------------------------
# baseline expected for 85EM block is optional; benchmark blocks are invoice-total based later
df["Underpayment_Dollar_85EM"] = 0.0
if "Expected_Amount_85_EM_invoice_level" in df.columns:
    df["Underpayment_Dollar_85EM"] = np.maximum(df["Expected_Amount_85_EM_invoice_level"] - df["Payment_Amount"], 0.0)

# Try to detect a reason/category column if present
reason_cols = [c for c in df.columns if re.search(r"(reason|category|denial|underpay)", c, flags=re.I)]
underpay_reason_col = reason_cols[0] if reason_cols else "Underpayment_Reason"
if underpay_reason_col not in df.columns:
    df[underpay_reason_col] = np.where(df["Underpayment_Dollar_85EM"]>0, "Rate Gap (Proxy: 85% E/M)", "None")

group_cols = ["Year","Week","Payer","Group_EM","Group_EM2", underpay_reason_col]
underpay_summary = (
    df.groupby(group_cols, dropna=False)
      .agg(Underpayment_Dollar_Sum=("Underpayment_Dollar_85EM","sum"),
           Invoices=("Invoice_Number","nunique"))
      .reset_index()
)
underpay_summary = underpay_summary.sort_values(["Year","Week","Underpayment_Dollar_Sum"],
                                                ascending=[True,True,False])

# --- Persist -----------------------------------------------------------------
df.to_csv(OUT_INVOICE, index=False)
underpay_summary.to_csv(OUT_UNDERPAY, index=False)

with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(OUT_INVOICE, arcname=OUT_INVOICE.name)
    z.write(OUT_UNDERPAY, arcname=OUT_UNDERPAY.name)

print("✅ 01_preprocess_and_enhance complete (invoice-centric; Visit_Count removed)")
print(f"   • Invoice index  -> {OUT_INVOICE}")
print(f"   • Underpayment   -> {OUT_UNDERPAY}")
print(f"   • Bundle         -> {OUT_ZIP}")
