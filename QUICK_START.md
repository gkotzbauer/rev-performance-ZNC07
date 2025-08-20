# 1) Clone your repo
git clone https://github.com/<your-org-or-user>/rev-performance-ZNC07.git
cd rev-performance-ZNC07

# 2) (Recommended) Create a clean virtual environment
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) (Optional) Set knobs (env vars) – see "Configuration Knobs" section for details
export DATA_DIR="/mnt/data"
export EXCLUDE_LATEST_FROM_BENCH="1"
export BACKTEST_K="6"

# 5) Run the pipeline (core)
python 01_preprocess_and_enhance.py
python 02_weekly_insights.py

# 6) (Recommended optional) NRV gap benchmarks + merge back into weekly
python 03_nrv_gap_benchmarks.py
python 03b_merge_nrv_gaps_into_weekly.py
