#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
master_pipeline.py — One-command runner for the Revenue Performance pipeline.

Pipeline (minimal → full):
  1) 01_preprocess_and_enhance.py              (required)
  2) 02_weekly_insights.py                     (required)
  3) 03_nrv_gap_benchmarks.py                  (optional; recommended, controlled by RUN_NRV_GAPS)
  4) 03b_merge_nrv_gaps_into_weekly.py         (optional; controlled by RUN_MERGE_NRV; depends on step 3)

Key environment knobs (tune without code edits):
  DATA_DIR=/mnt/data               Working dir for I/O (default: /mnt/data)

  # Weekly insights / selection engine
  EXCLUDE_LATEST_FROM_BENCH=1      Exclude latest Year-Week from history (1/0)
  BACKTEST_K=6                     Rolling backtest target weeks per key
  EWMA_ALPHA=0.2                   EWMA responsiveness (0.2 standard; 0.4 = faster)
  WMAPE_SELECTION=0                If 1, WMAPE is primary selection metric (else SMAPE)
  N_MIN_INVOICES=40                Min historical invoices to accumulate before scoring
  SHRINK_LAMBDA=20                 Empirical-Bayes prior strength (in invoice units)
  HAMPEL_K=3.0                     Hampel z-threshold for week-level outlier removal

  # Winsorization (weekly selection history)
  WINSOR_L=0.05                    Lower winsor bound (N>=20)
  WINSOR_U=0.95                    Upper winsor bound (N>=20)
  WINSOR_L_SMALL=0.10              Lower bound for small-N
  WINSOR_U_SMALL=0.90              Upper bound for small-N

  # Optional NRV gap modules
  RUN_NRV_GAPS=1                   If 1, run 03_nrv_gap_benchmarks.py
  RUN_MERGE_NRV=1                  If 1, run 03b_merge_nrv_gaps_into_weekly.py (requires step 3 outputs)

Usage:
  python master_pipeline.py
  python master_pipeline.py "Step 2"      # start at a label prefix (e.g., "Step 2")
"""

import os
import sys
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
DATA_DIR = Path(os.getenv("DATA_DIR", "/mnt/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------- Script Table ----------
SCRIPTS = [
    ("Step 1: Preprocess & Enhance",     "01_preprocess_and_enhance.py",       False),  # required
    ("Step 2: Weekly Insights",          "02_weekly_insights.py",              False),  # required
    ("Step 3: NRV Gap Benchmarks",       "03_nrv_gap_benchmarks.py",           True),   # optional via RUN_NRV_GAPS
    ("Step 3b: Merge NRV Gaps → Weekly", "03b_merge_nrv_gaps_into_weekly.py",  True),   # optional via RUN_MERGE_NRV
]

# ---------- CLI: optional start-at filter ----------
START_AT = None
if len(sys.argv) > 1:
    START_AT = sys.argv[1].strip().lower()

def should_run(label: str) -> bool:
    if START_AT is None:
        return True
    return label.strip().lower().startswith(START_AT)

# ---------- Pretty notes ----------
def _print_key_notes():
    print("\n🔧 CONFIG (env)")
    print(f"• DATA_DIR = {DATA_DIR}")
    print(f"• EXCLUDE_LATEST_FROM_BENCH = {os.getenv('EXCLUDE_LATEST_FROM_BENCH', '1')}")
    print(f"• BACKTEST_K = {os.getenv('BACKTEST_K', '6')} | EWMA_ALPHA = {os.getenv('EWMA_ALPHA', '0.2')}")
    print(f"• Selection primary = {'WMAPE' if os.getenv('WMAPE_SELECTION','0')=='1' else 'SMAPE'}")
    print(f"• N_MIN_INVOICES = {os.getenv('N_MIN_INVOICES', '40')} | SHRINK_LAMBDA = {os.getenv('SHRINK_LAMBDA','20')} | HAMPEL_K = {os.getenv('HAMPEL_K','3.0')}")
    print(f"• Winsor (N>=20): L={os.getenv('WINSOR_L','0.05')} U={os.getenv('WINSOR_U','0.95')} | small-N L={os.getenv('WINSOR_L_SMALL','0.10')} U={os.getenv('WINSOR_U_SMALL','0.90')}")
    print(f"• RUN_NRV_GAPS = {os.getenv('RUN_NRV_GAPS','1')} | RUN_MERGE_NRV = {os.getenv('RUN_MERGE_NRV','1')}\n")

# ---------- Preflights ----------
def _preflight_inputs_for_step1():
    # 01_preprocess_and_enhance.py can accept any of these as inputs; we just log what exists
    candidates = [
        DATA_DIR / "preprocess_invoice_data_output.xlsx",
        DATA_DIR / "preprocess_invoice_data_output.csv",
        DATA_DIR / "invoice_level_index_enhanced.xlsx",
        DATA_DIR / "RMT Invoice_level_index.xlsx",
    ]
    found = [p for p in candidates if p.is_file()]
    if not found:
        print("⚠️  Preflight (Step 1): no typical input files found; the script may still generate from raw sources if it has its own loaders.")
    else:
        print("✓ Preflight (Step 1): found candidate input(s):")
        for p in found:
            print("   •", p)

def _preflight_outputs_for_step2():
    # Step 2 typically reads the enhanced invoice-level data
    enh_csv = DATA_DIR / "enchance_invoice_metrics_output.csv"
    if not enh_csv.is_file():
        print(f"⚠️  Preflight (Step 2): {enh_csv.name} not present yet. This is expected before Step 1 runs.")
    else:
        print(f"✓ Preflight (Step 2): found {enh_csv.name}")

def _preflight_outputs_for_step3b():
    # Step 3b merges NRV gap outputs back into weekly file
    weekly_csv = DATA_DIR / "v2_Rev_Perf_Weekly_Model_Output_Final_granular.csv"
    nrv_invoice_csv = DATA_DIR / "nrv_gaps_by_invoice.csv"
    nrv_cpt_csv     = DATA_DIR / "nrv_gaps_by_cpt_cluster.csv"
    missing = []
    for p in [weekly_csv, nrv_invoice_csv, nrv_cpt_csv]:
        if not p.is_file():
            missing.append(p.name)
    if missing:
        print("⚠️  Preflight (Step 3b): missing required input(s):", ", ".join(missing))
    else:
        print("✓ Preflight (Step 3b): all NRV merge inputs present.")

# ---------- Run helper ----------
def _run(label: str, script: str, optional: bool, extra_env=None, notes: str=None):
    sp = HERE / script
    if not sp.is_file():
        if optional:
            print(f"⚠️  {label}: missing {script} — skipping (optional).")
            return
        raise FileNotFoundError(f"{label}: required script not found: {sp}")
    print(f"\n▶ {label} — running: {script}")
    if notes:
        print(notes)
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    env.setdefault("DATA_DIR", str(DATA_DIR))
    subprocess.run([sys.executable, str(sp)], check=True, env=env, cwd=str(HERE))
    print(f"✅ {label} — completed.")

def main():
    _print_key_notes()

    # Unified env overrides propagated to all children
    env_overrides = {
        "DATA_DIR": str(DATA_DIR),

        # weekly-insights / selection knobs
        "EXCLUDE_LATEST_FROM_BENCH": os.getenv("EXCLUDE_LATEST_FROM_BENCH", "1"),
        "BACKTEST_K": os.getenv("BACKTEST_K", "6"),
        "EWMA_ALPHA": os.getenv("EWMA_ALPHA", "0.2"),
        "WMAPE_SELECTION": os.getenv("WMAPE_SELECTION", "0"),
        "N_MIN_INVOICES": os.getenv("N_MIN_INVOICES", "40"),
        "SHRINK_LAMBDA": os.getenv("SHRINK_LAMBDA", "20"),
        "HAMPEL_K": os.getenv("HAMPEL_K", "3.0"),
        "WINSOR_L": os.getenv("WINSOR_L", "0.05"),
        "WINSOR_U": os.getenv("WINSOR_U", "0.95"),
        "WINSOR_L_SMALL": os.getenv("WINSOR_L_SMALL", "0.10"),
        "WINSOR_U_SMALL": os.getenv("WINSOR_U_SMALL", "0.90"),

        # optional NRV gap toggles
        "RUN_NRV_GAPS": os.getenv("RUN_NRV_GAPS", "1"),
        "RUN_MERGE_NRV": os.getenv("RUN_MERGE_NRV", "1"),
    }

    # Preflights
    _preflight_inputs_for_step1()
    _preflight_outputs_for_step2()

    # Decide optional steps
    run_nrv_gaps  = os.getenv("RUN_NRV_GAPS", "1") == "1"
    run_merge_nrv = os.getenv("RUN_MERGE_NRV", "1") == "1"

    for label, script, optional in SCRIPTS:
        if not should_run(label):
            print(f"⏭  Skipping {label} (start-at filter: '{START_AT}')")
            continue

        if "NRV Gap Benchmarks" in label and not run_nrv_gaps:
            print("⏭  Skipping Step 3 (RUN_NRV_GAPS=0).")
            continue
        if "Merge NRV Gaps" in label and not run_merge_nrv:
            print("⏭  Skipping Step 3b (RUN_MERGE_NRV=0).")
            continue

        if "Merge NRV Gaps" in label:
            _preflight_outputs_for_step3b()

        _run(label, script, optional=optional, extra_env=env_overrides)

    print("\n🎉 Pipeline finished.")

if __name__ == "__main__":
    main()
