#!/bin/bash
#SBATCH -J sigma0_stability
#SBATCH -p mcdermott
#SBATCH -t 0-24:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH -o logs/sigma0_stability_%j.out
#SBATCH -e logs/sigma0_stability_%j.err

source activate /orcd/data/jhm/001/om2/gelbanna/miniconda3/envs/asr312
cd /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory || exit 1

NB_IN=notebooks/2026-02-12_isi0-small-tests-on-model-sigma0-stability.ipynb
NB_OUT=notebooks/2026-02-12_isi0-small-tests-on-model-sigma0-stability_executed.ipynb

echo "=== Running sigma0 stability notebook ==="
echo "Start: $(date)"

jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3 \
  --output "$(basename $NB_OUT)" \
  "$NB_IN"

echo "End: $(date)"
echo "Output: $NB_OUT"
