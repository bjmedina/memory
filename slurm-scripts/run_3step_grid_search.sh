#!/bin/bash
#SBATCH -J 3step_grid_t5
#SBATCH -p mit_normal_gpu
#SBATCH -t 0-1:30:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
## sigma0 mode: one job per sigma0 value (20 values).
## Each job sweeps all 20x20=400 (sigma1, sigma2) combos.
#SBATCH --array=0-19
#SBATCH -o slurm-scripts/logs/%x_%A_%a.out
#SBATCH -e slurm-scripts/logs/%x_%A_%a.err

# =============================
# ENVIRONMENT SETUP
# =============================
source activate /orcd/data/jhm/001/gelbanna/miniconda3/envs/asr_312
cd /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory || exit 1

# =============================
# CONFIGURABLE PARAMETERS
# (override via env vars when submitting)
#
# Examples:
#   sbatch slurm-scripts/run_3step_grid_search.sh                    # broad geomspace, sigma0 mode
#   N_MC=50 ISIS="0 2 16" sbatch slurm-scripts/run_3step_grid_search.sh
#   T_STEP=8 sbatch slurm-scripts/run_3step_grid_search.sh
#
# Custom grids (set --array to match number of sigma0 values - 1):
#   SIGMA0_GRID="0.0 0.5 1.0" SIGMA1_GRID="0.0 0.1 0.2" SIGMA2_GRID="0.0 0.05 0.1" \
#     sbatch --array=0-2 slurm-scripts/run_3step_grid_search.sh  # 3 sigma0 values
# =============================

N_MC=${N_MC:-10}
ISIS="${ISIS:-0 1 2 4 8 16 32 64}"
PARALLEL_MODE="${PARALLEL_MODE:-sigma0}"
SAVE_DIR="${SAVE_DIR:-reports/figures/3step_grid_search_t5}"
SEED="${SEED:-43}"
METRIC="${METRIC:-cosine}"
WHICH_TASK="${WHICH_TASK:-0}"
ENCODER="${ENCODER:-resnet50}"
LAYER="${LAYER:-layer4}"
DEVICE="${DEVICE:-cuda}"
T_STEP="${T_STEP:-5}"
N_SEQUENCES="${N_SEQUENCES:-120}"
SEQ_LENGTH="${SEQ_LENGTH:-50}"
MIN_PAIRS="${MIN_PAIRS:-5}"

# Custom grids (optional; override defaults when set).
SIGMA0_GRID="${SIGMA0_GRID:-}"
SIGMA1_GRID="${SIGMA1_GRID:-}"
SIGMA2_GRID="${SIGMA2_GRID:-}"

echo "======================================="
echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "N_MC               = $N_MC"
echo "ISIS               = $ISIS"
echo "PARALLEL_MODE      = $PARALLEL_MODE"
echo "METRIC             = $METRIC"
echo "WHICH_TASK         = $WHICH_TASK"
echo "ENCODER            = $ENCODER"
echo "LAYER              = $LAYER"
echo "SEED               = $SEED"
echo "T_STEP             = $T_STEP"
echo "SAVE_DIR           = $SAVE_DIR"
[[ -n "$SIGMA0_GRID" ]] && echo "SIGMA0_GRID        = $SIGMA0_GRID"
[[ -n "$SIGMA1_GRID" ]] && echo "SIGMA1_GRID        = $SIGMA1_GRID"
[[ -n "$SIGMA2_GRID" ]] && echo "SIGMA2_GRID        = $SIGMA2_GRID"
echo "======================================="

# =============================
# EXECUTION
# =============================

GRID_ARGS=()
if [[ -n "$SIGMA0_GRID" || -n "$SIGMA1_GRID" || -n "$SIGMA2_GRID" ]]; then
    [[ -n "$SIGMA0_GRID" ]] && GRID_ARGS+=(--sigma0-grid $SIGMA0_GRID)
    [[ -n "$SIGMA1_GRID" ]] && GRID_ARGS+=(--sigma1-grid $SIGMA1_GRID)
    [[ -n "$SIGMA2_GRID" ]] && GRID_ARGS+=(--sigma2-grid $SIGMA2_GRID)
fi

python src/model/run_3step_grid_search.py \
    --job-index "$SLURM_ARRAY_TASK_ID" \
    --parallel-mode "$PARALLEL_MODE" \
    "${GRID_ARGS[@]}" \
    --n-mc "$N_MC" \
    --isis $ISIS \
    --seed "$SEED" \
    --metric "$METRIC" \
    --t-step "$T_STEP" \
    --n-sequences "$N_SEQUENCES" \
    --seq-length "$SEQ_LENGTH" \
    --min-pairs-per-isi "$MIN_PAIRS" \
    --which-task "$WHICH_TASK" \
    --is-multi \
    --encoder-type "$ENCODER" \
    --layer "$LAYER" \
    --device "$DEVICE" \
    --save-dir "$SAVE_DIR" \
    --resume

echo "Done: job_index=$SLURM_ARRAY_TASK_ID"
