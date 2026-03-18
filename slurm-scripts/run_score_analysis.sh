#!/bin/bash
#SBATCH -J score_analysis
#SBATCH -p mcdermott
#SBATCH -t 0-24:00:00
#SBATCH -n 4
#SBATCH -c 4
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH -o logs/score_analysis_%j.out
#SBATCH -e logs/score_analysis_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bjmedina@mit.edu


source activate /orcd/data/jhm/001/om2/gelbanna/miniconda3/envs/asr312
cd /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory || exit 1

# NB_IN=notebooks/2026-02-19_sigma2-stability.ipynb
# NB_OUT=notebooks/2026-02-19_sigma2-stability_executed.ipynb

#2026-02-17_plotting-scores-and-such.ipynb

NB_IN=notebooks/2026-02-17_plotting-scores-and-such.ipynb
# Timestamp for output naming
TS="$(date +'%Y-%m-%d_%H%M%S')"

# Make executed notebook name include timestamp
BASE_IN="$(basename "$NB_IN" .ipynb)"
NB_OUT="notebooks/executed/${BASE_IN}_executed-${TS}.ipynb"

echo "Will save to: $NB_OUT"


echo "=== Running score notebook ==="
echo "Start: $(date)"

jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3 \
  --output "$(basename $NB_OUT)" \
  "$NB_IN"

echo "End: $(date)"
echo "Output: $NB_OUT"
