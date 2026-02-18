#!/usr/bin/env bash
#SBATCH -J figs
#SBATCH -p normal
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH -o logs/PCSfigs_%A_%a.out
#SBATCH -e logs/PCSfigs_%A_%a.err
#SBATCH --array=0-99

#fsource ~/.bashrc
source activate /om2/user/gelbanna/miniconda3/envs/asr312

cd /om2/user/bjmedina/auditory-memory/memory/slurm-scripts
idx=$((SLURM_ARRAY_TASK_ID - 1))

python ../figures/generate-distancemodel-256PCs-graphs.py --param_idx "$idx" --noise_std 0.1
python ../figures/generate-distancemodel-256PCs-graphs.py --param_idx "$idx" --noise_std 0.2
python ../figures/generate-distancemodel-256PCs-graphs.py --param_idx "$idx" --noise_std 0.3
python ../figures/generate-distancemodel-256PCs-graphs.py --param_idx "$idx" --noise_std 0.4
python ../figures/generate-distancemodel-256PCs-graphs.py --param_idx "$idx" --noise_std 0.5
python ../figures/generate-distancemodel-256PCs-graphs.py --param_idx "$idx" --noise_std 0.6
python ../figures/generate-distancemodel-256PCs-graphs.py --param_idx "$idx" --noise_std 0.7
python ../figures/generate-distancemodel-256PCs-graphs.py --param_idx "$idx" --noise_std 0.8
python ../figures/generate-distancemodel-256PCs-graphs.py --param_idx "$idx" --noise_std 0.9
python ../figures/generate-distancemodel-256PCs-graphs.py --param_idx "$idx" --noise_std 1.0

python ../figures/generate-distancemodel-256PCs-graphs.py --param_idx "$idx" --noise_std 0.15
python ../figures/generate-distancemodel-256PCs-graphs.py --param_idx "$idx" --noise_std 0.25
python ../figures/generate-distancemodel-256PCs-graphs.py --param_idx "$idx" --noise_std 0.35
python ../figures/generate-distancemodel-256PCs-graphs.py --param_idx "$idx" --noise_std 0.45
python ../figures/generate-distancemodel-256PCs-graphs.py --param_idx "$idx" --noise_std 0.55
python ../figures/generate-distancemodel-256PCs-graphs.py --param_idx "$idx" --noise_std 0.65
python ../figures/generate-distancemodel-256PCs-graphs.py --param_idx "$idx" --noise_std 0.75
python ../figures/generate-distancemodel-256PCs-graphs.py --param_idx "$idx" --noise_std 0.85
python ../figures/generate-distancemodel-256PCs-graphs.py --param_idx "$idx" --noise_std 0.95
python ../figures/generate-distancemodel-256PCs-graphs.py --param_idx "$idx" --noise_std 0.05