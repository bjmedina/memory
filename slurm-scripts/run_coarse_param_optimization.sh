#!/bin/bash
#SBATCH -J coarse-param-opt
#SBATCH -p mcdermott
#SBATCH -t 0-12:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH -o logs/coarse-param-opt_%j.out
#SBATCH -e logs/coarse-param-opt_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bryanmaildina@gmail.com

source activate /orcd/data/jhm/001/om2/gelbanna/miniconda3/envs/asr312
cd /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory || exit 1

NB_IN="notebooks/2026-02-27_coarse-param-optimization.ipynb"

# Timestamp for output naming
TS="$(date +'%Y-%m-%d_%H%M%S')"
BASE_IN="$(basename "$NB_IN" .ipynb)"
NB_OUT="notebooks/executed/${BASE_IN}-${TS}.ipynb"

mkdir -p notebooks/executed
mkdir -p logs

echo "Will save to: $NB_OUT"
echo "=== Running coarse parameter optimization ==="
echo "Start: $(date)"

jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3 \
  --output "$(basename $NB_OUT)" \
  "$NB_IN"

echo "End: $(date)"
echo "Output: $NB_OUT"
