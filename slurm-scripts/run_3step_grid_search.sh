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
#   JOB_INDEX = GLOBAL_BASE + SLURM_ARRAY_TASK_ID
# GLOBAL_BASE / OFFSET / CHUNK_SIZE: set by submit_3step_grid_search_in_chunks.sh via sbatch --export=...
# Fallback (manual run): OFFSET=0 → JOB_INDEX = SLURM_ARRAY_TASK_ID
CHUNK_SIZE=150
OFFSET=4
TASK_LOCAL="$SLURM_ARRAY_TASK_ID"

JOB_INDEX=$(( OFFSET * CHUNK_SIZE + TASK_LOCAL ))                                                                                                                                                                              

# =============================
# CONFIGURABLE PARAMETERS
# =============================

PARALLEL_MODE="flat"
METRIC="cosine"
T_STEP=5
N_MC=5
SAVE_DIR="/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/reports/figures/3step_grid_search_metric-${METRIC}_t${T_STEP}_nmc${N_MC}"

echo "======================================="
echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "OFFSET              = $OFFSET"
echo "CHUNK_SIZE          = $CHUNK_SIZE"
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
    --n-mc "$N_MC" 

echo "Done: job_index=$JOB_INDEX"