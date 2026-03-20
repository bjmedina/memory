#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# Merge per-triple .npz files into a single grid_search_results_vec.npz.
# Run after the vectorized grid search array job completes.
#
# Usage:
#   # Submit with dependency on the array job:
#   JOB_ID=$(sbatch --parsable slurm-scripts/run_2d_grid_search_chunked.sh)
#   sbatch --dependency=afterok:$JOB_ID slurm-scripts/gather_2d_grid_search_vectorized.sh
#
#   # Or run manually:
#   bash slurm-scripts/gather_2d_grid_search_vectorized.sh
# ──────────────────────────────────────────────────────────────────────
#SBATCH -J 2d_grid_gather_vec
#SBATCH -p mit_normal_gpu
#SBATCH -t 0-0:10:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -o slurm-scripts/logs/%x_%j.out
#SBATCH -e slurm-scripts/logs/%x_%j.err

conda activate /orcd/data/jhm/001/bjmedina/miniconda3/envs/asr_312_312
cd /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory || exit 1

SAVE_DIR="${SAVE_DIR:-reports/figures/2d_grid_search_vectorized_dense}"

echo "Merging results from: $SAVE_DIR"

python src/model/run_2d_grid_search_vectorized.py \
    --merge \
    --save-dir "$SAVE_DIR"
