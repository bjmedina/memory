#!/bin/bash
#SBATCH -J mem_model_grid_singleISI
#SBATCH -p mcdermott  
#SBATCH -t 0-8:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH -o /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/slurm-scripts/logs/%x_%A_%a.out
#SBATCH -e /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/slurm-scripts/logs/%x_%A_%a.err

# =============================
# ENVIRONMENT SETUP
# =============================
source activate /orcd/data/jhm/001/om2/gelbanna/miniconda3/envs/asr312
cd /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/utls || exit 1

# =============================
# EXECUTION
# =============================
python3 /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/src/model/main-singleisi.py \
  --save_base "/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/figures/model-behavior_v9/"

echo "✅ Done: metric=$metric, noise_mode=$noise_mode, rate=$rate"