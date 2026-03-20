#!/bin/bash
# Submits 3-step grid search in batches, chaining with SLURM dependencies.
# Each batch waits for the previous one to finish before starting.
#
# Usage: bash slurm-scripts/submit_3step_batches.sh

TOTAL_JOBS=3375
BATCH_SIZE=150
PREV=""

for (( OFF=0; OFF<TOTAL_JOBS; OFF+=BATCH_SIZE )); do
    LEFT=$((TOTAL_JOBS - OFF))
    MAX=$(( (LEFT < BATCH_SIZE ? LEFT : BATCH_SIZE) - 1 ))

    DEP=""
    [[ -n "$PREV" ]] && DEP="--dependency=afterany:${PREV}"

    PREV=$(sbatch --array="0-${MAX}" --export="ALL,OFFSET=${OFF}" \
           $DEP slurm-scripts/run_3step_grid_search.sh | awk '{print $NF}')

    echo "Batch OFFSET=$OFF array=0-${MAX} jobid=$PREV"
done
