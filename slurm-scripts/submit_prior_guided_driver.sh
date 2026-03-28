#!/bin/bash
#SBATCH -J prior_driver
#SBATCH -p ou_bcs_normal
#SBATCH -t 0-24:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=1G
#SBATCH -o prior_driver.out
#SBATCH -e prior_driver.err

# Driver job: submits prior-guided grid search in batches and waits for
# each to finish before submitting the next.
#
# Usage:
#   sbatch slurm-scripts/submit_prior_guided_driver.sh
#
#   # With auto-merge after all batches:
#   GATHER=1 sbatch slurm-scripts/submit_prior_guided_driver.sh
#
#   # Override parameters:
#   METRIC=euclidean N_MC=5 WHICH_TASK=0 sbatch slurm-scripts/submit_prior_guided_driver.sh

conda activate /orcd/data/jhm/001/bjmedina/miniconda3/envs/asr_312_312
cd /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory || exit 1

# Build flags from environment
FLAGS=""
[[ "${GATHER:-0}" == "1" ]] && FLAGS="$FLAGS --gather"
[[ -n "$METRIC" ]]          && FLAGS="$FLAGS --metric $METRIC"
[[ -n "$N_MC" ]]            && FLAGS="$FLAGS --n-mc $N_MC"
[[ -n "$WHICH_TASK" ]]      && FLAGS="$FLAGS --which-task $WHICH_TASK"
[[ -n "$ENCODER" ]]         && FLAGS="$FLAGS --encoder $ENCODER"
[[ -n "$PC_DIMS" ]]         && FLAGS="$FLAGS --pc-dims $PC_DIMS"
[[ -n "$PARALLEL_MODE" ]]   && FLAGS="$FLAGS --parallel-mode $PARALLEL_MODE"
[[ -n "$SAVE_DIR" ]]        && FLAGS="$FLAGS --save-dir $SAVE_DIR"

echo "Running: python slurm-scripts/submit_prior_guided_batches.py $FLAGS"
python slurm-scripts/submit_prior_guided_batches.py $FLAGS
