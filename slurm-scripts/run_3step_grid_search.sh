#!/bin/bash
#SBATCH -J 3step_grid_t5
#SBATCH -p mit_normal_gpu
#SBATCH -t 0-1:30:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
## Array range and OFFSET are set by submit_3step_batches.sh.
## To run standalone: sbatch --array=0-14 slurm-scripts/run_3step_grid_search.sh
#SBATCH -o slurm-scripts/logs/%x_%A_%a.out
#SBATCH -e slurm-scripts/logs/%x_%A_%a.out

# =============================
# ENVIRONMENT SETUP
# =============================
conda activate /orcd/data/jhm/001/bjmedina/miniconda3/envs/asr_312_312
cd /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory || exit 1

# =============================
# CONFIGURABLE PARAMETERS (commented out — all use Python defaults now)
# Uncomment and pass as env vars to override, e.g.:
#   N_MC=50 sbatch slurm-scripts/run_3step_grid_search.sh
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

OFFSET=${OFFSET:-0}
BATCH_SIZE=${BATCH_SIZE:-150}
JOB_INDEX=$(( OFFSET * BATCH_SIZE + SLURM_ARRAY_TASK_ID ))

echo "======================================="
echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "OFFSET              = $OFFSET"
echo "BATCH_SIZE          = $BATCH_SIZE"
echo "JOB_INDEX           = $JOB_INDEX"
echo "======================================="

# =============================
# EXECUTION
# =============================

python src/model/run_3step_grid_search.py \
    --job-index "$JOB_INDEX" \
    --parallel-mode flat \
    --resume

echo "Done: job_index=$JOB_INDEX"
