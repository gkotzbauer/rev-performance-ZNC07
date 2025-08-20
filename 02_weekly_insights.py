#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
02_weekly_insights.py  —  Slim-2 drop-in

What this script does (key points):
- Builds weekly aggregates by key and computes Payment_Rate.
- Selects a leakage-free benchmark rate per row from {mean, median, winsorized-median},
  chosen by rolling backtest error (SMAPE primary; set WMAPE_SELECTION=1 to use WMAPE).
- Week-1 (no-history) fallback: Exp85 per-visit if available, else same-week cross-sectional
  median rate for (Payer, Group_EM, Group_EM2).
- Carries Exp85 per-visit through as a separate baseline and shows BOTH baselines on Summary:
  Expected (Selected) and Expected (Exp85), with % and $ variances for both.
- Drivers & Top Ups/Downs prefer Selected baseline; narratives include % and $ and de-duplicate.
- Restores _Avg baselines by (Payer, Group_EM, Group_EM2) for RC narratives.
- Adds parity check for selected benchmark rate vs implied rate.
- Exports:
    /mnt/data/Weekly_Performance_With_Diagnostics.xlsx
    /mnt/data/weekly_backtest_summary.csv
    /mnt/data/weekly_insights_bundle.zip
  Optional “lite” artifacts when WRITE_LITE=1:
    /mnt/data/Weekly_Performance_Summary_Lite.xlsx
    /mnt/data/All_Rows.csv.gz
    /mnt/data/weekly_insights_bundle_lite.zip
"""

import os
import numpy as np
import pandas as pd
import zipfile
from pathlib import Path

# ======================
# Config (env-tunable)
# ======================
DATA_DIR = Path(os.getenv("DATA_DIR", "/mnt/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

IN_INVOICE   = DATA_DIR / "invoice_level_index_enhanced.csv"
IN_UNDERPAY  = DATA_DIR / "underpayment_summary.csv"

OUT_XLSX     = DATA_DIR / "Weekly_Performance_With_Diagnostics.xlsx"
OUT_BACKTEST = DATA_DIR / "weekly_backtest_summary.csv"
OUT_ZIP      = DATA_DIR / "weekly_insights_bundle.zip"

# Optional slim exports
WRITE_LITE         = os.getenv("WRITE_LITE", "0") == "1"
OUT_XLSX_LITE      = DATA_DIR / "Weekly_Performance_Summary_Lite.xlsx"
OUT_ALL_ROWS_GZ    = DATA_DIR / "All_Rows.csv.gz"
OUT_ZIP_LITE       = DATA_DIR / "weekly_insights_bundle_lite.zip"

# Performance bands
PERF_OVER  = float(os.getenv("PERF_OVER_PCT",  "0.05"))
PERF_UNDER = float(os.getenv("PERF_UNDER_PCT", "-0.05"))

# Selection / backtest knobs
BACKTEST_K        = int(os.getenv("BACKTEST_K", "6"))       # rows/horizon per key for rolling CV
WMAPE_SELECTION   = os.getenv("WMAPE_SELECTION", "0") == "1"
WINSOR_L          = float(os.getenv("WINSOR_L", "0.05"))
WINSOR_U          = float(os.getenv("WINSOR_U", "0.95"))

# Parity tolerance for selected benchmark rate
BENCH_RATE_EPS    = float(os.getenv("BENCH_RATE_EPS", "0.01"))

# Narrative thresholds
DRIVER_MIN_ABS_PCT = float(os.getenv("DRIVER_MIN_ABS_PCT", "0.02"))   # 2%-pt vs selected
DRIVER_MIN_ABS_USD = float(os.getenv("DRIVER_MIN_ABS_USD", "500"))    # $500 abs delta

# ======================
# Helpers
# ======================
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

def winsorize(arr, lo=WINSOR_L, hi=WINSOR_U):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0: return arr
    ql = np.quantile(arr, lo); qh = np.quantile(arr, hi)
    return np.clip(arr, ql, qh)

def classify_label(actual, expected):
    if pd.isna(actual) or pd.isna(expected) or expected == 0: return "No Data"
    diff = (actual - expected) / expected
    if diff > PERF_OVER:  return "Over Performing"
    if diff < PERF_UNDER: return "Under Performing"
    return "Average Performance"

def fmt_pct(x): return "" if pd.isna(x) else f"{x:+.0%}"
def fmt_dol(x):
    if pd.isna(x): return ""
    val = f"{x:+,.0f}"
    return val

def candidate_from_history(hist_rates, method):
    hist = np.asarray(hist_rates, dtype=float)
    hist = hist[~np.isnan(hist)]
    if hist.size == 0: return np.nan
    if method == "mean":           return float(np.nanmean(hist))
    if method == "median":         return float(np.nanmedian(hist))
    if method == "winsor_median":
        if hist.size < 20:         return np.nan
        return float(np.nanmedian(winsorize(hist)))
    return np.nan

def rolling_cv_error(rates):
    methods = ["winsor_median", "median", "mean"]
    errs = {m: [] for m in methods}
    rates = list(rates); T = len(rates)
    start = max(1, T - BACKTEST_K)
    for t in range(1, T):
        if t < start: continue
        hist = rates[:t]; y = rates[t]
        preds = {m: candidate_from_history(hist, m) for m in methods}
        for m, p in preds.items():
            if np.isnan(p) or np.isnan(y):
                e = np.nan
            else:
                if WMAPE_SELECTION:
                    e = abs(y - p) / max(abs(y), 1e-9)                  # WMAPE-like
                else:
                    denom = (abs(y) + abs(p)) / 2.0                      # SMAPE
                    e = abs(y - p) / (denom if denom != 0 else np.nan)
            errs[m].append(e)
    avg = {m: (float(np.nanmean(v)) if np.isfinite(np.nanmean(v)) else 1e9) for m, v in errs.items()}
    return avg

def select_method(errs: dict):
    # tie-break priority favors robustness: winsor_median > median > mean
    prio = ["winsor_median", "median", "mean"]
    return min(errs, key=lambda m: (errs[m], prio.index(m)))

def variance_block(df: pd.DataFrame, expected_col: str, tag: str):
    dcol = f"Revenue_Variance_{tag}_$"
    pcol = f"Revenue_Variance_{tag}_%"
    lcol = f"Performance_Label_{tag}"
    if expected_col not in df.columns:
        df[dcol] = np.nan; df[pcol] = np.nan; df[lcol] = "No Data"; return
    df[dcol] = df["Payment_Amount"] - df[expected_col]
    df[pcol] = np.where(df[expected_col] == 0, np.nan, df[dcol] / df[expected_col])
    df[lcol] = df.apply(lambda r: classify_label(r["Payment_Amount"], r.get(expected_col, np.nan)), axis=1)

# ======================
# Load inputs
# ======================
if not IN_INVOICE.is_file():
    raise FileNotFoundError(f"Missing input: {IN_INVOICE}")
df = pd.read_csv(IN_INVOICE, low_memory=False)
underpay = pd.read_csv(IN_UNDERPAY) if IN_UNDERPAY.is_file() else pd.DataFrame()

# Normalize core types
for c in ["Year", "Week"]:
    if c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
for c in ["Payment_Amount", "Visit_Count", "Expected_Amount_85_EM_invoice_level"]:
    if c in df.columns:
        df[c] = df[c].apply(to_float_safe)
if "Visit_Count" not in df.columns or df["Visit_Count"].isna().all():
    df["Visit_Count"] = 1.0
for c in ["Payer","Group_EM","Group_EM2","CPT_List_Str","Benchmark_Key"]:
    if c not in df.columns: df[c] = ""

# ======================
# Weekly aggregation (+ carry Exp85 per-visit)
# ======================
gcols = ["Year","Week","Payer","Group_EM","Group_EM2","CPT_List_Str","Benchmark_Key"]
agg_map = {
    "Payment_Amount": "sum",
    "Visit_Count": "sum",
    "Expected_Amount_85_EM_invoice_level": "mean",  # per-visit expected (invoice-level metric)
    "Charge_Billed_Balance": "sum",
    "Zero_Balance_Collection_Star_Charges": "sum",
    "NRV_Zero_Balance": "sum",
    "Denial_Percent": "mean",
    "Zero_Balance_Collection_Rate": "mean",
    "Collection_Rate": "mean",
    "NRV_Gap_Dollar": "sum",
    "NRV_Gap_Percent": "mean",
    "NRV_Gap_Sum_Dollar": "sum",
}
present = {k: v for k, v in agg_map.items() if k in df.columns}
weekly = df.groupby(gcols, dropna=False).agg(present).reset_index()

weekly["Payment_Rate"] = np.where(weekly["Visit_Count"] > 0, weekly["Payment_Amount"] / weekly["Visit_Count"], np.nan)
if "Expected_Amount_85_EM_invoice_level" in weekly.columns:
    weekly["Exp85_per_visit"] = weekly["Expected_Amount_85_EM_invoice_level"]
    weekly["Exp85_Expected_Payment"] = weekly["Exp85_per_visit"] * weekly["Visit_Count"]
else:
    weekly["Exp85_per_visit"] = np.nan
    weekly["Exp85_Expected_Payment"] = np.nan

# Same-week cross-sectional fallback rate for (Payer, Group_EM, Group_EM2)
xsec = (weekly.groupby(["Year","Week","Payer","Group_EM","Group_EM2"], dropna=False)["Payment_Rate"]
              .median().rename("XSec_Median_Rate").reset_index())
weekly = weekly.merge(xsec, on=["Year","Week","Payer","Group_EM","Group_EM2"], how="left")

# ======================
# Benchmark selection (mean / median / winsor_median)
# ======================
weekly = weekly.sort_values(gcols).reset_index(drop=True)

def compute_selected_for_group(g: pd.DataFrame) -> pd.DataFrame:
    rates  = g["Payment_Rate"].to_numpy(dtype=float)
    visits = g["Visit_Count"].to_numpy(dtype=float)
    n = len(g)

    sel_rate   = np.full(n, np.nan)
    sel_pay    = np.full(n, np.nan)
    sel_method = np.full(n, None, dtype=object)

    mean_rate   = np.full(n, np.nan)
    median_rate = np.full(n, np.nan)
    winsor_rate = np.full(n, np.nan)

    mean_pay   = np.full(n, np.nan)
    median_pay = np.full(n, np.nan)
    winsor_pay = np.full(n, np.nan)

    for i in range(n):
        hist = rates[:i]
        r_mean   = candidate_from_history(hist, "mean")
        r_median = candidate_from_history(hist, "median")
        r_wmed   = candidate_from_history(hist, "winsor_median")

        mean_rate[i], median_rate[i], winsor_rate[i] = r_mean, r_median, r_wmed
        mean_pay[i]   = r_mean   * visits[i] if visits[i] > 0 and not np.isnan(r_mean)   else np.nan
        median_pay[i] = r_median * visits[i] if visits[i] > 0 and not np.isnan(r_median) else np.nan
        winsor_pay[i] = r_wmed   * visits[i] if visits[i] > 0 and not np.isnan(r_wmed)   else np.nan

        if i == 0 or np.sum(~np.isnan(hist)) == 0:
            # Week-1 fallback: Exp85 per-visit, else cross-sectional median
            fallback = g["Exp85_per_visit"].iloc[i]
            method   = "Fallback_Exp85"
            if pd.isna(fallback):
                fallback = g["XSec_Median_Rate"].iloc[i]
                method   = "Fallback_XSec"
            sel_method[i] = method
            sel_rate[i]   = fallback if not pd.isna(fallback) else np.nan
            sel_pay[i]    = fallback * visits[i] if not pd.isna(fallback) and visits[i] > 0 else np.nan
        else:
            errs = rolling_cv_error(hist)
            m = select_method(errs)
            sel_method[i] = m
            r_sel = {"mean": r_mean, "median": r_median, "winsor_median": r_wmed}[m]
            sel_rate[i] = r_sel
            sel_pay[i]  = r_sel * visits[i] if not np.isnan(r_sel) and visits[i] > 0 else np.nan

    g = g.copy()
    g["Selected_Benchmark_Rate"] = sel_rate
    g["Selected_Benchmark_Payment"] = sel_pay
    g["Benchmark_Method_Used"] = sel_method

    # explicit lenses
    g["Benchmark_Payment_Rate_per_Visit"]         = mean_rate
    g["Benchmark_Payment_Rate_per_Visit_Median"]  = median_rate
    g["Benchmark_Payment_Rate_per_Visit_Winsorized"] = winsor_rate

    g["Benchmark_Payment_Mean"]       = mean_pay
    g["Benchmark_Payment_Median"]     = median_pay
    g["Benchmark_Payment_Winsorized"] = winsor_pay
    return g

weekly = weekly.groupby(["Payer","Group_EM","Group_EM2","CPT_List_Str","Benchmark_Key"],
                        dropna=False, group_keys=False).apply(compute_selected_for_group)

# Selected baseline + lens variances
weekly["Expected_Payment"] = weekly["Selected_Benchmark_Payment"]
weekly["Revenue_Variance_vs_Selected_$"] = weekly["Payment_Amount"] - weekly["Expected_Payment"]
weekly["Revenue_Variance_vs_Selected_%"] = np.where(
    weekly["Expected_Payment"] == 0, np.nan,
    weekly["Revenue_Variance_vs_Selected_$"] / weekly["Expected_Payment"]
)
weekly["Performance_Label_vs_Selected"] = weekly.apply(
    lambda r: classify_label(r["Payment_Amount"], r["Expected_Payment"]), axis=1
)

for col, tag in [
    ("Selected_Benchmark_Payment", "vs_Benchmark_Selected"),
    ("Benchmark_Payment_Mean", "vs_Benchmark_Mean"),
    ("Benchmark_Payment_Median", "vs_Benchmark_Median"),
    ("Benchmark_Payment_Winsorized", "vs_Benchmark_Winsorized"),
]:
    variance_block(weekly, col, tag)

# _Avg baselines (payer × EM × EM2)
avg_metrics = [c for c in [
    "Charge_Billed_Balance",
    "Zero_Balance_Collection_Star_Charges",
    "NRV_Zero_Balance",
    "Zero_Balance_Collection_Rate",
    "Collection_Rate",
    "Payment_Amount",
    "Denial_Percent",
    "NRV_Gap_Dollar",
    "NRV_Gap_Percent",
    "Remaining_Charges_Percent",
    "NRV_Gap_Sum_Dollar",
] if c in weekly.columns]
if avg_metrics:
    base = (weekly.groupby(["Payer","Group_EM","Group_EM2"], dropna=False)[avg_metrics]
                  .mean(numeric_only=True)
                  .rename(columns={c: f"{c}_Avg" for c in avg_metrics})
                  .reset_index())
    weekly = weekly.merge(base, on=["Payer","Group_EM","Group_EM2"], how="left")

# Parity check for selected rate
weekly["_Benchmark_Implied_Rate"] = np.where(
    weekly["Visit_Count"] == 0, np.nan,
    weekly["Selected_Benchmark_Payment"] / weekly["Visit_Count"]
)
weekly["_Benchmark_Rate_Diff"] = weekly["_Benchmark_Implied_Rate"] - weekly["Selected_Benchmark_Payment_Rate_per_Visit"]
weekly["Benchmark_Rate_Parity_Flag"] = (weekly["_Benchmark_Rate_Diff"].abs() <= BENCH_RATE_EPS)

# ======================
# Drivers & Narratives
# ======================
bench_dollar_priority = [
    "Revenue_Variance_vs_Selected_$",
    "Revenue_Variance_vs_Benchmark_Median_$",
    "Revenue_Variance_vs_Benchmark_Winsorized_$",
    "Revenue_Variance_vs_Benchmark_Mean_$",
]

def pick_bench_delta(row):
    for c in bench_dollar_priority:
        if c in row and pd.notna(row[c]):
            return float(row[c]), c
    return np.nan, None

def build_row_driver(row):
    sel_pct = row.get("Revenue_Variance_vs_Selected_%", np.nan)
    sel_dol = row.get("Revenue_Variance_vs_Selected_$", np.nan)
    bench_dol, bench_col = pick_bench_delta(row)
    sig = (pd.notna(sel_pct) and abs(sel_pct) >= DRIVER_MIN_ABS_PCT) or \
          (pd.notna(bench_dol) and abs(bench_dol) >= DRIVER_MIN_ABS_USD)
    if not sig: return ""
    name = f"{row['Payer']} – {row['Group_EM']} – {row['Group_EM2']}"
    parts = []
    if pd.notna(sel_pct):
        parts.append(f"{fmt_pct(sel_pct)} vs selected ({fmt_dol(sel_dol)})")
    if pd.notna(bench_dol):
        tag = ("median Δ" if bench_col.endswith("Median_$")
               else "winsorized Δ" if bench_col.endswith("Winsorized_$")
               else "mean Δ")
        parts.append(f"{tag} {fmt_dol(bench_dol)}")
    return f"{name}: " + "; ".join(parts)

weekly["Benchmark Driver (Row)"] = weekly.apply(build_row_driver, axis=1)

# Week-level Drivers sheet narrative (top ~6)
drivers_rec = []
for (yr, wk), sub in weekly.groupby(["Year","Week"], dropna=False):
    sig_sub = sub.loc[sub["Benchmark Driver (Row)"].astype(str).str.len() > 0].copy()
    if not sig_sub.empty:
        sig_sub["_rank"] = -sig_sub["Revenue_Variance_vs_Selected_$"].abs().fillna(0)
        sig_sub = sig_sub.sort_values("_rank").drop(columns=["_rank"]).head(6)
        drivers_text = "; ".join(sig_sub["Benchmark Driver (Row)"].tolist())
    else:
        drivers_text = ""
    drivers_rec.append({"Year": yr, "Week": wk, "Benchmark Drivers Narrative": drivers_text})
drivers_df = pd.DataFrame(drivers_rec)

# ======================
# Summary (both baselines)
# ======================
wk_pay = weekly.groupby(["Year","Week"], dropna=False)["Payment_Amount"].sum().rename("Payments").reset_index()
wk_sel = weekly.groupby(["Year","Week"], dropna=False)["Expected_Payment"].sum(min_count=1).rename("Expected (Selected)").reset_index()
wk_85  = weekly.groupby(["Year","Week"], dropna=False)["Exp85_Expected_Payment"].sum(min_count=1).rename("Expected (Exp85)").reset_index()

summary = (wk_pay.merge(wk_sel, on=["Year","Week"], how="left")
                .merge(wk_85,  on=["Year","Week"], how="left")
                .merge(drivers_df, on=["Year","Week"], how="left"))

summary["Overall Variance $ (Selected)"] = summary["Payments"] - summary["Expected (Selected)"]
summary["Overall Variance % (Selected)"] = np.where(
    summary["Expected (Selected)"].eq(0), np.nan,
    summary["Overall Variance $ (Selected)"] / summary["Expected (Selected)"]
)
summary["Overall Variance $ (Exp85)"] = summary["Payments"] - summary["Expected (Exp85)"]
summary["Overall Variance % (Exp85)"] = np.where(
    summary["Expected (Exp85)"].eq(0), np.nan,
    summary["Overall Variance $ (Exp85)"] / summary["Expected (Exp85)"]
)

# Top Ups/Downs strings (vs Selected)
def pack_moves(d):
    d = d[(d["Revenue_Variance_vs_Selected_%"].notna()) | (d["Revenue_Variance_vs_Selected_$"].notna())]
    if d.empty: return ""
    labels = (d["Payer"] + " – " + d["Group_EM"] + " – " + d["Group_EM2"] + " " +
              d["Revenue_Variance_vs_Selected_%"].map(fmt_pct) + " " +
              d["Revenue_Variance_vs_Selected_$"].map(fmt_dol)).drop_duplicates()
    return "; ".join(labels.tolist())

top_up = (weekly.sort_values(["Year","Week","Revenue_Variance_vs_Selected_$"], ascending=[True,True,False])
               .groupby(["Year","Week"]).head(3)
               .groupby(["Year","Week"]).apply(pack_moves).reset_index()
               .rename(columns={0: "Top Upsides (vs Selected)"}))
top_dn = (weekly.sort_values(["Year","Week","Revenue_Variance_vs_Selected_$"], ascending=[True,True,True])
               .groupby(["Year","Week"]).head(3)
               .groupby(["Year","Week"]).apply(pack_moves).reset_index()
               .rename(columns={0: "Top Downsides (vs Selected)"}))

summary = summary.merge(top_up, on=["Year","Week"], how="left").merge(top_dn, on=["Year","Week"], how="left")

# Underpayment totals (if provided)
if not underpay.empty:
    under_sum = (underpay.groupby(["Year","Week"], dropna=False)["Underpayment_Dollar_Sum"]
                        .sum().reset_index()
                        .rename(columns={"Underpayment_Dollar_Sum": "Underpayment_Total_$"}))
    summary = summary.merge(under_sum, on=["Year","Week"], how="left")
else:
    summary["Underpayment_Total_$"] = np.nan

# Leadership summary string
summary["Leadership Summary"] = (
    "Vs Selected " + summary["Overall Variance % (Selected)"].map(fmt_pct) +
    " (" + summary["Overall Variance $ (Selected)"].map(fmt_dol) + "); " +
    "Vs Exp85 " + summary["Overall Variance % (Exp85)"].map(fmt_pct) +
    " (" + summary["Overall Variance $ (Exp85)"].map(fmt_dol) + "). " +
    "Upsides: " + summary["Top Upsides (vs Selected)"].fillna("") + ". " +
    "Downsides: " + summary["Top Downsides (vs Selected)"].fillna("") + ". " +
    "Underpayment total: " + summary["Underpayment_Total_$"].map(lambda x: "" if pd.isna(x) else f"${x:,.0f}") + ". " +
    "Drivers: " + summary["Benchmark Drivers Narrative"].fillna("") + "."
)

# ======================
# Backtest summary (share of methods used)
# ======================
bt = (weekly.groupby(["Payer","Group_EM","Group_EM2","CPT_List_Str","Benchmark_Key","Benchmark_Method_Used"], dropna=False)
             .size().reset_index(name="Rows"))
bt["Share"] = bt.groupby(["Payer","Group_EM","Group_EM2","CPT_List_Str","Benchmark_Key"])["Rows"].apply(lambda s: s / s.sum())
bt.to_csv(OUT_BACKTEST, index=False)

# ======================
# Export workbook
# ======================
with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as xw:
    summary.to_excel(xw, sheet_name="Summary", index=False)
    drivers_df.to_excel(xw, sheet_name="Drivers", index=False)
    if not underpay.empty:
        underpay.to_excel(xw, sheet_name="Underpayment", index=False)
    weekly.to_excel(xw, sheet_name="All_Rows", index=False)
    bt.to_excel(xw, sheet_name="Backtest_Summary", index=False)

# Zip bundle
with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as z:
    for p in [OUT_XLSX, OUT_BACKTEST]:
        if Path(p).exists():
            z.write(p, arcname=Path(p).name)

print(f"✅ Weekly insights written:")
print(f"   • Workbook: {OUT_XLSX}")
print(f"   • Backtest: {OUT_BACKTEST}")
print(f"   • Bundle:   {OUT_ZIP}")

# ======================
# Optional slim exports
# ======================
if WRITE_LITE:
    # 1) Lite workbook with small sheets only
    with pd.ExcelWriter(OUT_XLSX_LITE, engine="openpyxl") as xw:
        summary.to_excel(xw, sheet_name="Summary", index=False)
        drivers_df.to_excel(xw, sheet_name="Drivers", index=False)
        if not underpay.empty:
            underpay.to_excel(xw, sheet_name="Underpayment", index=False)
        bt.to_excel(xw, sheet_name="Backtest_Summary", index=False)

    # 2) All_Rows as compressed CSV
    weekly.to_csv(OUT_ALL_ROWS_GZ, index=False, compression="gzip")

    # 3) Lite bundle
    with zipfile.ZipFile(OUT_ZIP_LITE, "w", zipfile.ZIP_DEFLATED) as z:
        for p in [OUT_XLSX_LITE, OUT_ALL_ROWS_GZ, OUT_BACKTEST]:
            if Path(p).exists():
                z.write(p, arcname=Path(p).name)

    print(f"📦 Lite artifacts:")
    print(f"   • Lite workbook: {OUT_XLSX_LITE}")
    print(f"   • All_Rows.gz:   {OUT_ALL_ROWS_GZ}")
    print(f"   • Lite bundle:   {OUT_ZIP_LITE}")
