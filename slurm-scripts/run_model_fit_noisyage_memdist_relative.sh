#!/usr/bin/env bash
#SBATCH -J memfit
#SBATCH -p normal
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH -o logs/noisyagememfit_%A_%a.out
#SBATCH -e logs/noisyagememfit_%A_%a.err
##SBATCH --array=0-19


#fsource ~/.bashrc
source activate /orcd/data/jhm/001/om2/gelbanna/miniconda3/envs/asr312

# Location of the driver
cd /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/slurm-scripts

python ../src/model/optimize_noisy_age_memory_dist_model_relative.py