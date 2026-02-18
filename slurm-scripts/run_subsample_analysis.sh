#!/bin/bash
#SBATCH -J mem_yaml_array
#SBATCH -p mcdermott
#SBATCH -t 0-20:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --array=0-4
#SBATCH -o logs/subsample_%a.out
#SBATCH -e logs/subsample_%a.err

source activate /om2/user/gelbanna/miniconda3/envs/asr312
cd /om2/user/bjmedina/auditory-memory/memory || exit 1

# Explicit parameter list
# PARAMS=(8 16 32 48 64)
PARAMS=(8 16 32 64 72)

# Select parameter for this task
SUBSAMPLE=${PARAMS[$SLURM_ARRAY_TASK_ID]}

echo "Running subsample_analysis with SUBSAMPLE=${SUBSAMPLE}"

python src/model/subsample_analysis.py ${SUBSAMPLE}


