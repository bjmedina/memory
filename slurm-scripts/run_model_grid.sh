#!/bin/bash
#SBATCH -J mem_model_grid
#SBATCH -p mcdermott  
#SBATCH -t 0-3:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -o /om2/user/bjmedina/auditory-memory/memory/slurm-scripts/logs/%x_%A_%a.out
#SBATCH -e /om2/user/bjmedina/auditory-memory/memory/slurm-scripts/logs/%x_%A_%a.err

# =============================
# ENVIRONMENT SETUP
# =============================
#fsource ~/.bashrc
source activate /om2/user/gelbanna/miniconda3/envs/asr312

cd /om2/user/bjmedina/auditory-memory/memory/utls || exit 1

# =============================
# PARAMETER GRID
# =============================

metrics=("loglikelihood" "euclidean")
noise_modes=("decay" "power-decay" "exp-decay" "const")
rates=(0.1 0.3 0.5 0.7)

# Total number of jobs = len(metrics) * len(noise_modes) * len(rates)
# e.g. 2 * 4 * 3 = 24
total_jobs=$(( ${#metrics[@]} * ${#noise_modes[@]} * ${#rates[@]} ))

# Safety check
if [ "$SLURM_ARRAY_TASK_ID" -ge "$total_jobs" ]; then
  echo "Task ID $SLURM_ARRAY_TASK_ID exceeds job grid size $total_jobs."
  exit 1
fi

# =============================
# INDEXING INTO GRID
# =============================

# Compute indices for each dimension
metric_idx=$(( SLURM_ARRAY_TASK_ID / (${#noise_modes[@]} * ${#rates[@]}) ))
noise_idx=$(( (SLURM_ARRAY_TASK_ID / ${#rates[@]}) % ${#noise_modes[@]} ))
rate_idx=$(( SLURM_ARRAY_TASK_ID % ${#rates[@]} ))

metric=${metrics[$metric_idx]}
noise_mode=${noise_modes[$noise_idx]}
rate=${rates[$rate_idx]}

echo "Running combination:"
echo "  Metric:     $metric"
echo "  Noise mode: $noise_mode"
echo "  Rate:       $rate"

# =============================
# EXECUTION
# =============================

python3 /om2/user/bjmedina/auditory-memory/memory/src/model/main.py \
  --metric "$metric" \
  --noise_mode "$noise_mode" \
  --rate "$rate" \
  --run_id "prolific_batch" \
  --save_base "/om2/user/bjmedina/auditory-memory/memory/figures/model-behavior/"

echo "✅ Done: metric=$metric, noise_mode=$noise_mode, rate=$rate"