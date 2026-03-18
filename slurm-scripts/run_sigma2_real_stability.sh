#!/bin/bash
#SBATCH -J sigma2_real_stability
#SBATCH -p mcdermott
#SBATCH -t 0-24:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH -o logs/sigma2_real_stability_%j.out
#SBATCH -e logs/sigma2_real_stability_%j.err

source activate /om2/user/gelbanna/miniconda3/envs/asr312
cd /om2/user/bjmedina/auditory-memory/memory || exit 1

NB_IN=notebooks/2026-02-23_sigma2-real-experiments-stability.ipynb
NB_OUT=notebooks/2026-02-23_sigma2-real-experiments-stability_executed.ipynb

echo "=== Running sigma2 real-experiments stability notebook ==="
echo "Start: $(date)"

jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3 \
  --output "$(basename $NB_OUT)" \
  "$NB_IN"

echo "End: $(date)"
echo "Output: $NB_OUT"
