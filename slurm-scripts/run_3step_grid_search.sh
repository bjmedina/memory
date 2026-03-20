#!/bin/bash
#SBATCH -J 3step_grid_t5
#SBATCH -p mit_normal_gpu
#SBATCH -t 0-0:30:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
## sigma0 mode: one job per sigma0 value (20 values).
## Each job sweeps all 20x20=400 (sigma1, sigma2) combos.
#SBATCH --array=0-19
#SBATCH -o slurm-scripts/logs/%x_%A_%a.out
#SBATCH -e slurm-scripts/logs/%x_%A_%a.out

# =============================
# ENVIRONMENT SETUP
# =============================
source activate /orcd/data/jhm/001/gelbanna/miniconda3/envs/asr_312
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

echo "======================================="
echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "======================================="

# =============================
# EXECUTION
# =============================

python src/model/run_3step_grid_search.py \
    --job-index "$SLURM_ARRAY_TASK_ID" \
    --resume

echo "Done: job_index=$SLURM_ARRAY_TASK_ID"
