#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03_nrv_gap_benchmarks.py

Purpose
-------
Pinpoint and summarize gaps where invoice-level Payment != NRV Zero Balance,
emitting TWO grains:

A) Invoice + CPT grouping:
   Year | Week | Payer | Group_EM | Group_EM2 | CPT_List_Str | Invoice_Number
   -> /mnt/data/nrv_gap_by_invoice.csv

B) CPT grouping (no invoice):
   Year | Week | Payer | Group_EM | Group_EM2 | CPT_List_Str
   -> /mnt/data/nrv_gap_by_cpt_cluster.csv

What’s a “gap”?
---------------
NRV_Payment_Gap_$ = Payment Amount* − NRV Zero Balance*
NRV_Payment_Gap_% = (Payment Amount* − NRV Zero Balance*) / max(NRV Zero Balance*, tiny)

We flag a gap when BOTH:
  • |NRV_Payment_Gap_$| >= GAP_TOLERANCE_ABS
  • |NRV_Payment_Gap_%| >= GAP_TOLERANCE_PCT

Knobs (env)
-----------
DATA_DIR               (default: /mnt/data)
INPUT_ENHANCED_PATH    (override full path to enhanced CSV/XLSX; default /mnt/data/enchance_invoice_metrics_output.csv)
GAP_TOLERANCE_ABS      (default: 0.00)  e.g., 1.00 to ignore tiny $ differences
GAP_TOLERANCE_PCT      (default: 0.00)  e.g., 0.005 for 0.5%
WINSOR_L               (default: 0.05)
WINSOR_U               (default: 0.95)
INCLUDE_WINS           (default: 1)     If 1, include winsorized mean gap benchmarks in cluster output

Inputs (must exist in enhanced file)
------------------------------------
- Year, Week, Payer, Group_EM, Group_EM2, CPT_List_Str, Invoice_Number
- Payment Amount*
- NRV Zero Balance*   (will be canonicalized from common variants if needed)

Outputs
-------
- /mnt/data/nrv_gap_by_invoice.csv
- /mnt/data/nrv_gap_by_cpt_cluster.csv
- /mnt/data/nrv_gap_benchmarks.xlsx (two sheets; optional best-effort)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# -----------------------------
# Config / paths
# -----------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", "/mnt/data"))
INPUT_ENHANCED_PATH = os.getenv("INPUT_ENHANCED_PATH", str(DATA_DIR / "enchance_invoice_metrics_output.csv"))

GAP_TOLERANCE_ABS = float(os.getenv("GAP_TOLERANCE_ABS", "0.00"))
GAP_TOLERANCE_PCT = float(os.getenv("GAP_TOLERANCE_PCT", "0.00"))

WINS_L   = float(os.getenv("WINSOR_L", "0.05"))
WINS_U   = float(os.getenv("WINSOR_U", "0.95"))
INCLUDE_WINS = int(os.getenv("INCLUDE_WINS", "1")) == 1

OUT_INVOICE   = str(DATA_DIR / "nrv_gap_by_invoice.csv")
OUT_CPT_GROUP = str(DATA_DIR / "nrv_gap_by_cpt_cluster.csv")
OUT_XLSX      = str(DATA_DIR / "nrv_gap_benchmarks.xlsx")

# -----------------------------
# Helpers
# -----------------------------
def _read_enhanced(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p, low_memory=False)
    if p.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(p)
    # fallback search
    csv_f = DATA_DIR / "enchance_invoice_metrics_output.csv"
    xlsx_f = DATA_DIR / "enchance_invoice_metrics_output.xlsx"
    if csv_f.is_file():
        return pd.read_csv(csv_f, low_memory=False)
    if xlsx_f.is_file():
        return pd.read_excel(xlsx_f)
    raise FileNotFoundError(f"Enhanced file not found: {path}")

def _canonize_nrv_zero_balance(df: pd.DataFrame) -> pd.DataFrame:
    candidates = [
        "NRV Zero Balance*", "NRV Zero Balance", "NRV_Zero_Balance", "NRV_Zero_Balance*",
        "NRV Zero Balance *"
    ]
    for c in candidates:
        if c in df.columns:
            if c != "NRV Zero Balance*":
                df.rename(columns={c: "NRV Zero Balance*"}, inplace=True)
            return df
    # if still missing, create NaN column (we can still run, but nothing will flag)
    df["NRV Zero Balance*"] = np.nan
    print("⚠️  'NRV Zero Balance*' not found; created empty column. Gaps will be NaN and no rows will flag.")
    return df

def _require_cols(df: pd.DataFrame, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

def _winsor_mean(x: pd.Series, l=0.05, u=0.95):
    s = pd.to_numeric(x, errors="coerce").dropna()
    if s.empty:
        return np.nan
    lo, hi = s.quantile([l, u])
    return s.clip(lo, hi).mean()

def _tiny():
    return 1e-9

def _normalize_key_types(df: pd.DataFrame) -> pd.DataFrame:
    for c in ["Year","Week"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in ["Payer","Group_EM","Group_EM2","CPT_List_Str","Benchmark_Key","Invoice_Number"]:
        if c in df.columns:
            df[c] = df[c].astype(str)
    return df

# -----------------------------
# Load & prepare
# -----------------------------
df = _read_enhanced(INPUT_ENHANCED_PATH)
df = _canonize_nrv_zero_balance(df)
_require_cols(df, [
    "Year","Week","Payer","Group_EM","Group_EM2","CPT_List_Str","Invoice_Number",
    "Payment Amount*","NRV Zero Balance*"
])

# Ensure numeric
for c in ["Payment Amount*","NRV Zero Balance*"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Basic grain normalization
df = _normalize_key_types(df)

# -----------------------------
# Compute gaps (line-level -> invoice view)
# -----------------------------
# Summarize to invoice granularity first to avoid double counting:
inv = (
    df.groupby(["Year","Week","Payer","Group_EM","Group_EM2","CPT_List_Str","Invoice_Number"], dropna=False)
      .agg(
          Payment_Total = ("Payment Amount*", "sum"),
          NRV_Total     = ("NRV Zero Balance*", "sum"),
          Lines         = ("Invoice_Number", "count")
      ).reset_index()
)

inv["NRV_Payment_Gap_$"] = inv["Payment_Total"] - inv["NRV_Total"]
inv["NRV_Payment_Gap_%"] = np.where(
    inv["NRV_Total"].abs() <= _tiny(), np.nan,
    inv["NRV_Payment_Gap_$"] / inv["NRV_Total"].replace(0, np.nan)
)

inv["Gap_Flag"] = (
    (inv["NRV_Payment_Gap_$"].abs() >= GAP_TOLERANCE_ABS) &
    (inv["NRV_Payment_Gap_%"].abs() >= GAP_TOLERANCE_PCT)
).astype(int)

# Save invoice+key output
inv_cols = [
    "Year","Week","Payer","Group_EM","Group_EM2","CPT_List_Str","Invoice_Number",
    "Payment_Total","NRV_Total","NRV_Payment_Gap_$","NRV_Payment_Gap_%","Gap_Flag","Lines"
]
inv[inv_cols].to_csv(OUT_INVOICE, index=False)

# -----------------------------
# Cluster (CPT grouping) output
# -----------------------------
grp_cols = ["Year","Week","Payer","Group_EM","Group_EM2","CPT_List_Str"]

def _agg_pos_mean(s: pd.Series):
    s = pd.to_numeric(s, errors="coerce")
    s = s[~s.isna()]
    s = s[s != 0]
    return s[s>0].mean() if (s>0).any() else 0.0

def _agg_neg_mean_abs(s: pd.Series):
    # average magnitude for negative gaps
    s = pd.to_numeric(s, errors="coerce")
    s = s[~s.isna()]
    s = s[s != 0]
    return (-s[s<0]).mean() if (s<0).any() else 0.0

cluster = (
    inv.groupby(grp_cols, dropna=False)
       .agg(
           Invoices=("Invoice_Number", "nunique"),
           Lines=("Lines", "sum"),
           Payment_Total=("Payment_Total","sum"),
           NRV_Total=("NRV_Total","sum"),
           Gap_Total_$=("NRV_Payment_Gap_$","sum"),
           Gap_Positive_Count=("NRV_Payment_Gap_$", lambda s: (s>0).sum()),
           Gap_Negative_Count=("NRV_Payment_Gap_$", lambda s: (s<0).sum()),
           Gap_Flagged_Invoices=("Gap_Flag","sum"),
           Gap_Positive_Avg_$=("NRV_Payment_Gap_$", _agg_pos_mean),
           Gap_Negative_AvgAbs_$=("NRV_Payment_Gap_$", _agg_neg_mean_abs),
       ).reset_index()
)

cluster["Gap_Rate_%Invoices"] = np.where(
    cluster["Invoices"]==0, np.nan, cluster["Gap_Flagged_Invoices"] / cluster["Invoices"]
)

# Cluster-level “gap benchmarks” (optional): mean/median and winsorized mean of invoice gaps in the cluster
gap_bench = (
    inv.groupby(grp_cols, dropna=False)["NRV_Payment_Gap_$"]
       .agg(Gap_Bench_Mean="mean", Gap_Bench_Median="median")
       .reset_index()
)

if INCLUDE_WINS:
    wins = inv.groupby(grp_cols, dropna=False)["NRV_Payment_Gap_$"] \
              .apply(lambda s: _winsor_mean(s, WINS_L, WINS_U)) \
              .rename("Gap_Bench_Winsorized").reset_index()
    gap_bench = gap_bench.merge(wins, on=grp_cols, how="left")
else:
    gap_bench["Gap_Bench_Winsorized"] = np.nan

cluster = cluster.merge(gap_bench, on=grp_cols, how="left")

# Save cluster output
cluster.to_csv(OUT_CPT_GROUP, index=False)

# -----------------------------
# Best-effort Excel (both sheets)
# -----------------------------
try:
    with pd.ExcelWriter(OUT_XLSX) as w:
        inv[inv_cols].to_excel(w, index=False, sheet_name="By_Invoice")
        cluster.to_excel(w, index=False, sheet_name="By_CPT_Cluster")
except Exception as e:
    print(f"Note: Excel export skipped ({e}); CSVs are written:\n - {OUT_INVOICE}\n - {OUT_CPT_GROUP}")

print("✅ NRV gap benchmarks emitted:")
print(f"   • Invoice + CPT grouping : {OUT_INVOICE}")
print(f"   • CPT grouping (cluster) : {OUT_CPT_GROUP}")
print(f"   • Tolerances: abs>={GAP_TOLERANCE_ABS:.4f}, pct>={GAP_TOLERANCE_PCT:.4%}; winsor={'on' if INCLUDE_WINS else 'off'}")
