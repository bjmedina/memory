#!/bin/bash
#SBATCH -J 2d_vec_driver
#SBATCH -p ou_bcs_low
#SBATCH -t 2-0:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -o slurm-scripts/logs/%x_%j.out
#SBATCH -e slurm-scripts/logs/%x_%j.out

# Driver job: submits 2D vectorized grid search in batches and waits for
# each to finish before submitting the next.  Run with:
#
#   sbatch slurm-scripts/submit_2d_vec_driver.sh                   # dense grid (default)
#   GRID=fine sbatch slurm-scripts/submit_2d_vec_driver.sh         # fine grid
#   GRID=dense GATHER=1 sbatch slurm-scripts/submit_2d_vec_driver.sh  # dense + auto-merge

cd /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory || exit 1
conda activate /orcd/data/jhm/001/bjmedina/miniconda3/envs/asr_312_312

GRID="${GRID:-dense}"
GATHER_FLAG=""
[[ "${GATHER:-0}" == "1" ]] && GATHER_FLAG="--gather"

python slurm-scripts/submit_2d_vec_batches.py --grid "$GRID" $GATHER_FLAG
