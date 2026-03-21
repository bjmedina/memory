#!/bin/bash
#SBATCH -J 3step_driver
#SBATCH -p ou_bcs_low
#SBATCH -t 1-0:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -o slurm-scripts/logs/%x_%j.out
#SBATCH -e slurm-scripts/logs/%x_%j.out

# Driver job: submits 3-step grid search in batches and waits for each to
# finish before submitting the next.  Run with:
#   sbatch slurm-scripts/submit_3step_driver.sh

cd /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory || exit 1

bash slurm-scripts/submit_3step_batches.sh
