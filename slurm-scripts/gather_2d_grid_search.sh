#!/bin/bash
# ──────────────────────────────────────────────────────────────────────
# Gather per-triple .npz files into master CSV + grid .npz.
# Run after the job array (run_2d_grid_search.sh) completes.
#
# Usage:
#   # Submit with dependency on the array job:
#   JOB_ID=$(sbatch --parsable slurm-scripts/run_2d_grid_search.sh)
#   sbatch --dependency=afterok:$JOB_ID slurm-scripts/gather_2d_grid_search.sh
#
#   # Or run manually:
#   bash slurm-scripts/gather_2d_grid_search.sh
# ──────────────────────────────────────────────────────────────────────
#SBATCH -J isi_gather
#SBATCH -p mit_normal_gpu
#SBATCH -t 0-0:10:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G
#SBATCH -o logs/isi_gather_%j.out
#SBATCH -e logs/isi_gather_%j.err

source activate /orcd/data/jhm/001/gelbanna/miniconda3/envs/asr_312
cd /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory || exit 1

OUTPUT_DIR="${OUTPUT_DIR:-reports/figures/2d_grid_search/slurm_$(date +'%Y%m%d')}"

echo "Gathering results from: $OUTPUT_DIR"

python scripts/run_2d_grid_search.py \
    --output-dir "$OUTPUT_DIR" \
    --gather
