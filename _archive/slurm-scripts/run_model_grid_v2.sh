#!/bin/bash
#SBATCH -J mem_model_grid
#SBATCH -p mcdermott  
#SBATCH -t 0-8:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH -o /om2/user/bjmedina/auditory-memory/memory/slurm-scripts/logs/%x_%A_%a.out
#SBATCH -e /om2/user/bjmedina/auditory-memory/memory/slurm-scripts/logs/%x_%A_%a.err

# =============================
# ENVIRONMENT SETUP
# =============================
source activate /om2/user/gelbanna/miniconda3/envs/asr312
cd /om2/user/bjmedina/auditory-memory/memory/utls || exit 1

# =============================
# PARAMETER GRID (filtered)
# =============================

metrics=("loglikelihood" "euclidean" "cosine")
#metrics=("cosine")
#noise_modes=("exp-decay")
noise_modes=("const-step-v2")
rates=(0 1 2 3 4 5 6 7 8 9)
#rates=(0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0)

# Build a list of valid triplets dynamically
combinations=()

for metric in "${metrics[@]}"; do
  for noise in "${noise_modes[@]}"; do
    case "$noise" in
      "exp-law"| "power-decay"| "exp-decay"| "power-law" | "linear-step" | "exp-step" |"exp-step-v2" | "linear-step-v2"|"const-step-v2")
        for rate in "${rates[@]}"; do
          combinations+=("$metric,$noise,$rate")
        done
        ;;
      "decay"|"const"|"const-step")
        # These don't use rate → assign rate=None
        combinations+=("$metric,$noise,0.0")
        ;;
    esac
  done
done

total_jobs=${#combinations[@]}

# Sanity check
if [ "$SLURM_ARRAY_TASK_ID" -ge "$total_jobs" ]; then
  echo "Task ID $SLURM_ARRAY_TASK_ID exceeds job grid size $total_jobs."
  exit 1
fi

# Parse this job's parameters
IFS=',' read -r metric noise_mode rate <<< "${combinations[$SLURM_ARRAY_TASK_ID]}"

echo "======================================="
echo "Running combination:"
echo "  Metric:     $metric"
echo "  Noise mode: $noise_mode"
echo "  Rate:       $rate"
echo "======================================="

# =============================
# EXECUTION
# =============================
python3 /om2/user/bjmedina/auditory-memory/memory/src/model/main.py \
  --metric "$metric" \
  --noise_mode "$noise_mode" \
  --rate "$rate" \
  --run_id "prolific_batch" \
  --save_base "/om2/user/bjmedina/auditory-memory/memory/figures/model-behavior-const-step-v4/"

echo "✅ Done: metric=$metric, noise_mode=$noise_mode, rate=$rate"