#!/bin/bash
#SBATCH -J 3step_grid
#SBATCH -p ou_bcs_low
#SBATCH -t 0-0:15:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
## Default array size; submit script overrides --array for the last (shorter) chunk.
#SBATCH --array=0-149
#SBATCH -o slurm-scripts/logs/%x_%A_%a.out
#SBATCH -e slurm-scripts/logs/%x_%A_%a.out

# =============================
# ENVIRONMENT SETUP
# =============================
conda activate /orcd/data/jhm/001/bjmedina/miniconda3/envs/asr_312_312
cd /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory || exit 1

# Flat index for Python:
#   JOB_INDEX = OFFSET * BATCH_SIZE + SLURM_ARRAY_TASK_ID
# OFFSET / BATCH_SIZE: set by submit_3step_batches.sh via sbatch --export=...
# Fallback (manual run): OFFSET=0 → JOB_INDEX = SLURM_ARRAY_TASK_ID

OFFSET=${OFFSET:-0}
BATCH_SIZE=150

# =============================
# CONFIGURABLE PARAMETERS
# =============================
# N_MC=${N_MC:-10}
# ISIS="${ISIS:-0 1 2 4 8 16 32 64}"
# PARALLEL_MODE="${PARALLEL_MODE:-sigma0}"
# SAVE_DIR="${SAVE_DIR:-reports/figures/3step_grid_search_t5}"
# SEED="${SEED:-43}"
# METRIC="${METRIC:-cosine}"
# WHICH_TASK="${WHICH_TASK:-0}"
# ENCODER="${ENCODER:-resnet50}"
# LAYER="${LAYER:-layer4}"
# DEVICE="${DEVICE:-cuda}"
# T_STEP="${T_STEP:-5}"
# N_SEQUENCES="${N_SEQUENCES:-100}"
# SEQ_LENGTH="${SEQ_LENGTH:-120}"
# MIN_PAIRS="${MIN_PAIRS:-5}"
# SIGMA0_GRID="${SIGMA0_GRID:-}"
# SIGMA1_GRID="${SIGMA1_GRID:-}"
# SIGMA2_GRID="${SIGMA2_GRID:-}"

JOB_INDEX=$(( OFFSET * BATCH_SIZE + SLURM_ARRAY_TASK_ID ))

PARALLEL_MODE="flat"
METRIC="cosine"
T_STEP=4
N_MC=1
SAVE_DIR="/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/reports/figures/3step_grid_search_metric-${METRIC}_t${T_STEP}_nmc${N_MC}_task0"

echo "======================================="
echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "OFFSET              = $OFFSET"
echo "BATCH_SIZE          = $BATCH_SIZE"
echo "JOB_INDEX           = $JOB_INDEX  (flat index for Python)"
echo "T_STEP              = $T_STEP"
echo "PARALLEL_MODE       = $PARALLEL_MODE"
echo "METRIC              = $METRIC"
echo "SAVE_DIR            = $SAVE_DIR"
echo "======================================="

# =============================
# EXECUTION
# =============================

python /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/src/model/run_3step_grid_search.py \
    --job-index "$JOB_INDEX" \
    --parallel-mode "$PARALLEL_MODE" \
    --t-step "$T_STEP" \
    --metric "$METRIC" \
    --save-dir "$SAVE_DIR" \
    --n-mc "$N_MC" \
    --resume

echo "Done: job_index=$JOB_INDEX"
