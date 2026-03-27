#!/bin/bash
#SBATCH -J prior_gather
#SBATCH -p ou_bcs_low
#SBATCH -t 0-0:10:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -o slurm-scripts/logs/%x_%A.out
#SBATCH -e slurm-scripts/logs/%x_%A.out

# Merge all per-triple/per-slice .npz files into a single result file.
#
# Usage:
#   sbatch slurm-scripts/gather_prior_guided_grid_search.sh
#
#   # With custom save dir:
#   SAVE_DIR=/path/to/results sbatch slurm-scripts/gather_prior_guided_grid_search.sh

cd /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory || exit 1
conda activate /orcd/data/jhm/001/bjmedina/miniconda3/envs/asr_312_312

METRIC="${METRIC:-euclidean}"
N_MC="${N_MC:-1}"
WHICH_TASK="${WHICH_TASK:-2}"
SAVE_DIR="${SAVE_DIR:-/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/reports/figures/prior_guided_grid_search_metric-${METRIC}_nmc${N_MC}_task${WHICH_TASK}}"

echo "Merging results from: $SAVE_DIR"

python src/model/run_prior_guided_grid_search.py \
    --merge \
    --save-dir "$SAVE_DIR"

echo "Done."
