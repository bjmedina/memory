#!/bin/bash
#SBATCH -J prior_grid
#SBATCH -p ou_bcs_low
#SBATCH -t 0-0:30:00
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
# OFFSET / BATCH_SIZE: set by submit script via sbatch --export=...
# Fallback (manual run): OFFSET=0 → JOB_INDEX = SLURM_ARRAY_TASK_ID

OFFSET=${OFFSET:-0}
BATCH_SIZE=150

# =============================
# CONFIGURABLE PARAMETERS
# =============================
JOB_INDEX=$(( OFFSET * BATCH_SIZE + SLURM_ARRAY_TASK_ID ))

PARALLEL_MODE="${PARALLEL_MODE:-flat}"
METRIC="${METRIC:-euclidean}"
N_MC="${N_MC:-1}"
WHICH_TASK="${WHICH_TASK:-2}"
ENCODER="${ENCODER:-texture_pca}"
PC_DIMS="${PC_DIMS:-256}"
DEVICE="${DEVICE:-cuda}"
SAVE_DIR="${SAVE_DIR:-/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/reports/figures/prior_guided_grid_search_metric-${METRIC}_task${WHICH_TASK}_nseq300_len135}"

echo "======================================="
echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "OFFSET              = $OFFSET"
echo "BATCH_SIZE          = $BATCH_SIZE"
echo "JOB_INDEX           = $JOB_INDEX  (flat index for Python)"
echo "PARALLEL_MODE       = $PARALLEL_MODE"
echo "METRIC              = $METRIC"
echo "N_MC                = $N_MC"
echo "WHICH_TASK          = $WHICH_TASK"
echo "ENCODER             = $ENCODER"
echo "PC_DIMS             = $PC_DIMS"
echo "SAVE_DIR            = $SAVE_DIR"
echo "======================================="

# =============================
# EXECUTION
# =============================

python /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/src/model/run_prior_guided_grid_search.py \
    --job-index "$JOB_INDEX" \
    --parallel-mode "$PARALLEL_MODE" \
    --metric "$METRIC" \
    --save-dir "$SAVE_DIR" \
    --n-mc "$N_MC" \
    --which-task "$WHICH_TASK" \
    --encoder-type "$ENCODER" \
    --pc-dims "$PC_DIMS" \
    --device "$DEVICE" \
    --resume

echo "Done: job_index=$JOB_INDEX"
