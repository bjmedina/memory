#!/usr/bin/env bash
#SBATCH -J figs
#SBATCH -p normal
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH -o logs/powerlawfigs_%A_%a.out
#SBATCH -e logs/powerlafigs_%A_%a.err
#SBATCH --array=0-8

#fsource ~/.bashrc
source activate /orcd/data/jhm/001/om2/gelbanna/miniconda3/envs/asr312

cd /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/slurm-scripts
idx=$((SLURM_ARRAY_TASK_ID - 1))
python ../figures/generate-distancemodel-powerlaw-graphs.py --param_idx "$idx"