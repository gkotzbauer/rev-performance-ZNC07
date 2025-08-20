#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
03b_merge_nrv_gaps_into_weekly.py

Purpose
-------
Attach NRV-vs-Payment gap signals to the weekly granular export without changing its grain.

Join logic
----------
Weekly grain (from 02_weekly_insights.py):
  Year | Week | Payer | Group_EM | Group_EM2 | Benchmark_Key

NRV gap cluster grain (from 03_nrv_gap_benchmarks.py):
  Year | Week | Payer | Group_EM | Group_EM2 | CPT_List_Str

We first aggregate the cluster table to:
  Year | Week | Payer | Group_EM | Group_EM2
then LEFT JOIN to the weekly file on those five keys.

Env knobs
---------
DATA_DIR            (default: /mnt/data)
WEEKLY_IN           (default: v2_Rev_Perf_Weekly_Model_Output_Final_granular.csv)
GAP_CLUSTER_IN      (default: nrv_gap_by_cpt_cluster.csv)
GAP_INVOICE_IN      (optional; default: nrv_gap_by_invoice.csv)  # not required to merge, used for diagnostics
WEEKLY_OUT_ENRICHED (default: v2_Rev_Perf_Weekly_Model_Output_Final_granular_enriched.csv)
WRITE_XLSX          (default: 1)  -> also writes ..._enriched.xlsx (best-effort)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(os.getenv("DATA_DIR", "/mnt/data"))
WEEKLY_IN = os.getenv("WEEKLY_IN", str(DATA_DIR / "v2_Rev_Perf_Weekly_Model_Output_Final_granular.csv"))
GAP_CLUSTER_IN = os.getenv("GAP_CLUSTER_IN", str(DATA_DIR / "nrv_gap_by_cpt_cluster.csv"))
GAP_INVOICE_IN = os.getenv("GAP_INVOICE_IN", str(DATA_DIR / "nrv_gap_by_invoice.csv"))
WEEKLY_OUT = os.getenv("WEEKLY_OUT_ENRICHED", str(DATA_DIR / "v2_Rev_Perf_Weekly_Model_Output_Final_granular_enriched.csv"))
WRITE_XLSX = int(os.getenv("WRITE_XLSX", "1")) == 1
WEEKLY_OUT_XLSX = WEEKLY_OUT.replace(".csv", ".xlsx")

def _req(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing required file: {path}")

_req(WEEKLY_IN)
_req(GAP_CLUSTER_IN)

weekly = pd.read_csv(WEEKLY_IN, low_memory=False)
cluster = pd.read_csv(GAP_CLUSTER_IN, low_memory=False)

# Normalize key types
for c in ["Year","Week"]:
    if c in weekly.columns: weekly[c] = pd.to_numeric(weekly[c], errors="coerce")
    if c in cluster.columns: cluster[c] = pd.to_numeric(cluster[c], errors="coerce")
for c in ["Payer","Group_EM","Group_EM2"]:
    if c in weekly.columns: weekly[c] = weekly[c].astype(str)
    if c in cluster.columns: cluster[c] = cluster[c].astype(str)

# Aggregate cluster from (Year,Week,Payer,Group_EM,Group_EM2,CPT_List_Str)
# -> (Year,Week,Payer,Group_EM,Group_EM2)
grp_cols5 = ["Year","Week","Payer","Group_EM","Group_EM2"]

def _safe_mean_pos(s):
    s = pd.to_numeric(s, errors="coerce")
    s = s[~s.isna()]
    s = s[s > 0]
    return s.mean() if len(s) else 0.0

def _safe_mean_abs_neg(s):
    s = pd.to_numeric(s, errors="coerce")
    s = s[~s.isna()]
    s = s[s < 0]
    return (-s).mean() if len(s) else 0.0

agg = (
    cluster.groupby(grp_cols5, dropna=False)
           .agg(
               NRVGap_Invoices=("Invoices","sum"),
               NRVGap_Lines=("Lines","sum"),
               NRVGap_Payment_Total=("Payment_Total","sum"),
               NRVGap_NRV_Total=("NRV_Total","sum"),
               NRVGap_Gap_Total_$=("Gap_Total_$","sum"),
               NRVGap_Gap_Positive_Count=("Gap_Positive_Count","sum"),
               NRVGap_Gap_Negative_Count=("Gap_Negative_Count","sum"),
               NRVGap_Gap_Flagged_Invoices=("Gap_Flagged_Invoices","sum"),
               NRVGap_Gap_Positive_Avg_$=("Gap_Positive_Avg_$","mean"),
               NRVGap_Gap_Negative_AvgAbs_$=("Gap_Negative_AvgAbs_$","mean"),
               NRVGap_Gap_Bench_Mean=("Gap_Bench_Mean","mean"),
               NRVGap_Gap_Bench_Median=("Gap_Bench_Median","median"),
               NRVGap_Gap_Bench_Winsorized=("Gap_Bench_Winsorized","mean"),
           ).reset_index()
)

# Rates derived at 5-key grain
agg["NRVGap_Gap_Rate_%Invoices"] = np.where(
    agg["NRVGap_Invoices"] == 0, np.nan, agg["NRVGap_Gap_Flagged_Invoices"] / agg["NRVGap_Invoices"]
)

# Merge onto weekly
join_keys = grp_cols5
enriched = weekly.merge(agg, on=join_keys, how="left")

# Write outputs
enriched.to_csv(WEEKLY_OUT, index=False)
if WRITE_XLSX:
    try:
        enriched.to_excel(WEEKLY_OUT_XLSX, index=False)
    except Exception as e:
        print(f"Note: Excel export skipped ({e}); CSV written: {WEEKLY_OUT}")

print("✅ Weekly file enriched with NRV gap measures")
print(f"   • Source weekly  : {WEEKLY_IN}")
print(f"   • Source gaps    : {GAP_CLUSTER_IN}")
print(f"   • Output (CSV)   : {WEEKLY_OUT}")
if WRITE_XLSX:
    print(f"   • Output (XLSX)  : {WEEKLY_OUT_XLSX}")
