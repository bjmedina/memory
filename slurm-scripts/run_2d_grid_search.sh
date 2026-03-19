#!/bin/bash
#SBATCH -J 2d_grid_search
#SBATCH -p mit_preemptable
#SBATCH -t 0-4:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
# Default: 8 jobs (one per sigma0). For fine grid use: sbatch --array=0-14 ... and set FINE_GRID=true
#SBATCH --array=0-14
#SBATCH -o slurm-scripts/logs/%x_%A_%a.out
#SBATCH -e slurm-scripts/logs/%x_%A_%a.err

# =============================
# ENVIRONMENT SETUP
# =============================
conda activate /orcd/data/jhm/001/gelbanna/miniconda3/envs/asr_312
cd /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory || exit 1

# =============================
# CONFIGURABLE PARAMETERS
# (override via env vars when submitting)
#
# Examples:
#   N_MC=50 ISIS="0 2 16" sbatch slurm-scripts/run_2d_grid_search.sh
#   PARALLEL_MODE=flat sbatch --array=0-391 slurm-scripts/run_2d_grid_search.sh
#   N_MC=100 METRIC=cosine SEED=123 sbatch slurm-scripts/run_2d_grid_search.sh
#
# Fine-grained grid (~2x resolution, 15×13×13 = 2535 triples, 15 sigma0 slices):
#   FINE_GRID=true sbatch --array=0-14 slurm-scripts/run_2d_grid_search.sh
# =============================

N_MC=${N_MC:-30}
ISIS="${ISIS:-0 2 8 16}"
PARALLEL_MODE="${PARALLEL_MODE:-sigma0}"
FINE_GRID="${FINE_GRID:-true}"
SAVE_DIR="${SAVE_DIR:-reports/figures/2d_grid_search_full}"
SEED="${SEED:-42}"
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

python src/model/run_2d_grid_search.py \
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
    --save-dir "$SAVE_DIR"

echo "Done: job_index=$SLURM_ARRAY_TASK_ID"
