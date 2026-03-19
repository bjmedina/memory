#!/bin/bash
#SBATCH -J 2d_grid_search_vec
#SBATCH -p mit_normal_gpu
#SBATCH -t 0-0:30:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G
## Flat mode: one job per (sigma0, sigma, eta) triple.
## Fine grid: 15 x 13 x 13 = 2535 triples → array 0-2534.
## %500 throttle limits concurrent jobs to avoid overwhelming the scheduler.
#SBATCH --array=0-2534%500
#SBATCH -o slurm-scripts/logs/%x_%A_%a.out
#SBATCH -e slurm-scripts/logs/%x_%A_%a.err

# =============================
# ENVIRONMENT SETUP
# =============================
# Use source activate (reliable in batch); conda activate often fails in non-interactive SLURM.
source activate /orcd/data/jhm/001/gelbanna/miniconda3/envs/asr_312
cd /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory || exit 1

# =============================
# CONFIGURABLE PARAMETERS
# (override via env vars when submitting)
#
# Output goes to 2d_grid_search_vectorized by default (separate from non-vectorized).
#
# Examples:
#   sbatch slurm-scripts/run_2d_grid_search_vectorized.sh          # fine grid, flat mode, 2535 jobs
#   N_MC=50 ISIS="0 2 16" sbatch slurm-scripts/run_2d_grid_search_vectorized.sh
#   N_MC=100 METRIC=cosine SEED=123 sbatch slurm-scripts/run_2d_grid_search_vectorized.sh
#
# sigma0 mode (fewer, longer jobs — one per sigma0 slice):
#   PARALLEL_MODE=sigma0 sbatch --array=0-14 slurm-scripts/run_2d_grid_search_vectorized.sh
#
# Local test (one task, same as SLURM would run for job 0):
#   cd /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory
#   source activate /orcd/data/jhm/001/gelbanna/miniconda3/envs/asr_312
#   SLURM_ARRAY_TASK_ID=0 bash slurm-scripts/run_2d_grid_search_vectorized.sh
# =============================

N_MC=${N_MC:-10}
ISIS="${ISIS:-0 2 8 16}"
PARALLEL_MODE="${PARALLEL_MODE:-flat}"
FINE_GRID="${FINE_GRID:-true}"
# Distinct from non-vectorized (2d_grid_search_full / 2d_grid_search)
SAVE_DIR="${SAVE_DIR:-reports/figures/2d_grid_search_vectorized_full}"
SEED="${SEED:-43}"
METRIC="${METRIC:-euclidean}"
N_SEQUENCES="${N_SEQUENCES:-100}"
SEQ_LENGTH="${SEQ_LENGTH:-99}"
MIN_PAIRS="${MIN_PAIRS:-5}"

echo "======================================="
echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "N_MC               = $N_MC"
echo "ISIS               = $ISIS"
echo "PARALLEL_MODE      = $PARALLEL_MODE"
echo "FINE_GRID          = $FINE_GRID"
echo "METRIC             = $METRIC"
echo "SEED               = $SEED"
echo "SAVE_DIR           = $SAVE_DIR"
echo "======================================="

# =============================
# EXECUTION
# =============================
FINE_ARGS=()
[[ "$FINE_GRID" == true ]] && FINE_ARGS=(--fine)

python src/model/run_2d_grid_search_vectorized.py \
    --job-index "$SLURM_ARRAY_TASK_ID" \
    --parallel-mode "$PARALLEL_MODE" \
    "${FINE_ARGS[@]}" \
    --n-mc "$N_MC" \
    --isis $ISIS \
    --seed "$SEED" \
    --metric "$METRIC" \
    --n-sequences "$N_SEQUENCES" \
    --seq-length "$SEQ_LENGTH" \
    --min-pairs-per-isi "$MIN_PAIRS" \
    --save-dir "$SAVE_DIR" \
    --resume

echo "Done: job_index=$SLURM_ARRAY_TASK_ID"
