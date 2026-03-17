#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# run_2d_grid_search.sh — Run the 3-way (σ₀, σ, η) parameter grid search
#
# Saves comprehensive per-triple data: raw scores, ROC curves, d', AUC,
# bootstrap SEM, hit/FA distributions — everything you need to look up
# any single (σ₀, σ, η) configuration after the fact.
#
# Usage:
#   bash scripts/run_2d_grid_search.sh                  # full run (392 triples)
#   bash scripts/run_2d_grid_search.sh --n-mc 2         # quick test (2 MC reps)
#   bash scripts/run_2d_grid_search.sh --resume         # continue interrupted run
#   bash scripts/run_2d_grid_search.sh --output-dir /my/dir
#
# Output structure:
#   <output-dir>/
#       grid_search_master.csv          # summary: one row per triple
#       grid_search_results.npz         # 3-D d' arrays (notebook-compat)
#       per_triple/                     # one .npz per triple (full raw data)
#
# Querying results for a specific triple:
#
#   python3 -c "
#   import pandas as pd, numpy as np
#   df = pd.read_csv('<output-dir>/grid_search_master.csv')
#   row = df[(df.sigma0 == 0.5) & (df.sigma == 0.1) & (df.eta == 0.02)]
#   print(row.T)
#
#   data = np.load('<output-dir>/per_triple/s0=0.500_sig=0.100_eta=0.020.npz',
#                  allow_pickle=True)
#   print('Hit scores (ISI=0):', data['hit_scores_isi0'])
#   print('ROC FPR (ISI=0):',    data['roc_fpr_isi0'])
#   print('d-prime (ISI=0):',    data['dprime_isi0'])
#   "
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

# Navigate to repo root (one level up from scripts/)
cd "$(dirname "$0")/.." || exit 1

TIMESTAMP="$(date +'%Y-%m-%d_%H%M%S')"
DEFAULT_OUTPUT="reports/figures/2d_grid_search/${TIMESTAMP}"

echo "═══════════════════════════════════════════════════"
echo "  2D Parameter Grid Search"
echo "═══════════════════════════════════════════════════"
echo "Start:   $(date)"
echo "Host:    $(hostname)"
echo "Dir:     $(pwd)"
echo ""

# Run the Python script, passing through all CLI arguments.
# If no --output-dir is given, use a timestamped default.
HAS_OUTPUT_DIR=false
for arg in "$@"; do
    if [[ "$arg" == "--output-dir" ]]; then
        HAS_OUTPUT_DIR=true
        break
    fi
done

if $HAS_OUTPUT_DIR; then
    python3 scripts/run_2d_grid_search.py "$@"
else
    echo "Output:  ${DEFAULT_OUTPUT}"
    echo ""
    python3 scripts/run_2d_grid_search.py --output-dir "${DEFAULT_OUTPUT}" "$@"
fi

echo ""
echo "═══════════════════════════════════════════════════"
echo "  Finished: $(date)"
echo "═══════════════════════════════════════════════════"
