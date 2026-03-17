#!/bin/bash
#SBATCH -J 2d_grid_search
#SBATCH -p mcdermott
#SBATCH -t 0-4:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH --array=0-7
#SBATCH -o slurm-scripts/logs/%x_%A_%a.out
#SBATCH -e slurm-scripts/logs/%x_%A_%a.err

# =============================
# ENVIRONMENT SETUP
# =============================
source activate /om2/user/gelbanna/miniconda3/envs/asr312
cd /om2/user/bjmedina/auditory-memory/memory || exit 1

# =============================
# CONFIGURABLE PARAMETERS
# (override via env vars when submitting)
#
# Examples:
#   N_MC=50 ISIS="0 2 16" sbatch slurm-scripts/run_2d_grid_search.sh
#   PARALLEL_MODE=flat sbatch --array=0-391 slurm-scripts/run_2d_grid_search.sh
#   N_MC=100 METRIC=cosine SEED=123 sbatch slurm-scripts/run_2d_grid_search.sh
# =============================

N_MC=${N_MC:-10}
ISIS="${ISIS:-0 2 16}"
PARALLEL_MODE="${PARALLEL_MODE:-sigma0}"
SAVE_DIR="${SAVE_DIR:-reports/figures/2d_grid_search}"
SEED="${SEED:-42}"
METRIC="${METRIC:-euclidean}"
N_SEQUENCES="${N_SEQUENCES:-10}"
SEQ_LENGTH="${SEQ_LENGTH:-81}"
MIN_PAIRS="${MIN_PAIRS:-5}"

echo "======================================="
echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "N_MC               = $N_MC"
echo "ISIS               = $ISIS"
echo "PARALLEL_MODE      = $PARALLEL_MODE"
echo "METRIC             = $METRIC"
echo "SEED               = $SEED"
echo "SAVE_DIR           = $SAVE_DIR"
echo "======================================="

# =============================
# EXECUTION
# =============================
python src/model/run_2d_grid_search.py \
    --job-index "$SLURM_ARRAY_TASK_ID" \
    --parallel-mode "$PARALLEL_MODE" \
    --n-mc "$N_MC" \
    --isis $ISIS \
    --seed "$SEED" \
    --metric "$METRIC" \
    --n-sequences "$N_SEQUENCES" \
    --seq-length "$SEQ_LENGTH" \
    --min-pairs-per-isi "$MIN_PAIRS" \
    --save-dir "$SAVE_DIR"

echo "Done: job_index=$SLURM_ARRAY_TASK_ID"
