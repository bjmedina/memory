#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# Merge per-triple .npz files into a single grid_search_results_3step_t5.npz.
# Run after the 3-step grid search array job completes.
#
# Usage:
#   # Submit with dependency on the array job:
#   JOB_ID=$(sbatch --parsable slurm-scripts/run_3step_grid_search.sh)
#   sbatch --dependency=afterok:$JOB_ID slurm-scripts/gather_3step_grid_search.sh
#
#   # Or run manually:
#   bash slurm-scripts/gather_3step_grid_search.sh
# ──────────────────────────────────────────────────────────────────────
#SBATCH -J 3step_grid_gather_t5
#SBATCH -p mit_normal_gpu
#SBATCH -t 0-0:10:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -o slurm-scripts/logs/%x_%j.out
#SBATCH -e slurm-scripts/logs/%x_%j.err

conda activate /orcd/data/jhm/001/bjmedina/miniconda3/envs/asr_312_312
cd /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory || exit 1

SAVE_DIR="/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/reports/figures/3step_grid_search_metric-cosine_t5_nmc5"

echo "Merging results from: $SAVE_DIR"

python src/model/run_3step_grid_search.py \
    --merge \
    --save-dir "$SAVE_DIR"
