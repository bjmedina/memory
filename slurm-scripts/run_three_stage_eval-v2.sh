#!/bin/bash
#SBATCH -J three-stage-compact-eval-v2
#SBATCH -p mcdermott
#SBATCH -t 0-25:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH -o logs/three-stage-compact-eval-v2_%j.out
#SBATCH -e logs/three-stage-compact-eval-v2_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bryanmaildina@gmail.com

source activate /om2/user/gelbanna/miniconda3/envs/asr312
cd /om2/user/bjmedina/auditory-memory/memory || exit 1

# NB_IN=notebooks/2026-02-19_sigma2-stability.ipynb
# NB_OUT=notebooks/2026-02-19_sigma2-stability_executed.ipynb

#2026-02-25_sigma2-compact-sequences.ipynb

NB_IN="notebooks/2026-02-26_three-stage-compact-eval-v2.ipynb"

# Timestamp for output naming
TS="$(date +'%Y-%m-%d_%H%M%S')"

# Make executed notebook name include timestamp
BASE_IN="$(basename "$NB_IN" .ipynb)"
NB_OUT="notebooks/executed/${BASE_IN}-${TS}.ipynb"

echo "Will save to: $NB_OUT"


echo "=== Running three-stage-compact-eval v2 notebook ==="
echo "Start: $(date)"

jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3 \
  --output "$(basename $NB_OUT)" \
  "$NB_IN"

echo "End: $(date)"
echo "Output: $NB_OUT"
