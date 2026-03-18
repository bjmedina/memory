#!/usr/bin/env bash
#SBATCH -J memfit
#SBATCH -p normal
#SBATCH -t 24:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem=250G
#SBATCH --gres=gpu:1
#SBATCH -o logs/memfit_%A_%a.out
#SBATCH -e logs/memfit_%A_%a.err
##SBATCH --array=0-19


#fsource ~/.bashrc
source activate /orcd/data/jhm/001/om2/gelbanna/miniconda3/envs/asr312
# Kept as CLI args (not hardcoded in code)
# WHICH_TASK="atexts-len120"
# RESULTS_ROOT="/mindhive/mcdermott/www/bjmedina/experiments/bolivia_2025/results/isi_16"
# SEQ_BASE="/mindhive/mcdermott/www/mturk_stimuli/bjmedina/{task}/sequences/isi_16/len120"
# OUT_DIR="/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/figures/human-results/isi-16-only/${WHICH_TASK}"
# METHOD="Powell"
# MIN_DPRIME=2.0
# ZSCORE_FIT_GLOB="/mindhive/mcdermott/www/mturk_stimuli/bjmedina/mem_exp_atexts_p1/*.wav"
# # Toggle skip_len60 by including the flag:
# SKIP_LEN60="--skip_len60"

# MIN_TRIALS=120
# DEVICE="gpu"
# EXP_SUBSET=2
# SEED=123

# Location of the driver
cd /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/slurm-scripts

python ../src/model/optimize_memory_model.py