#!/bin/bash
#SBATCH -J isi_sweep
#SBATCH -p mcdermott
#SBATCH -t 0-1:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH --array=0-391
#SBATCH -o logs/isi_sweep_%A_%a.out
#SBATCH -e logs/isi_sweep_%A_%a.err

source activate /om2/user/gelbanna/miniconda3/envs/asr312
cd /om2/user/bjmedina/auditory-memory/memory || exit 1

# All array tasks write to the same output directory
OUTPUT_DIR="${OUTPUT_DIR:-reports/figures/2d_grid_search/slurm_$(date +'%Y%m%d')}"

echo "======================================="
echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "OUTPUT_DIR = $OUTPUT_DIR"
echo "======================================="

python scripts/run_2d_grid_search.py \
    --output-dir "$OUTPUT_DIR" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --num-tasks 392 \
    --resume
