#!/bin/bash
# Submits 3-step grid search in batches, waiting for each to finish before
# submitting the next. This avoids QOSMaxSubmitJobPerUserLimit errors.
#
# Usage: bash slurm-scripts/submit_3step_batches.sh

TOTAL_JOBS=3375
BATCH_SIZE=150
NUM_BATCHES=$(( (TOTAL_JOBS + BATCH_SIZE - 1) / BATCH_SIZE ))
POLL_INTERVAL=60  # seconds between squeue checks

for (( BATCH=0; BATCH<NUM_BATCHES; BATCH++ )); do
    START=$(( BATCH * BATCH_SIZE ))
    LEFT=$((TOTAL_JOBS - START))
    MAX=$(( (LEFT < BATCH_SIZE ? LEFT : BATCH_SIZE) - 1 ))

    JOBID=$(sbatch --array="0-${MAX}" \
            --export="ALL,OFFSET=${BATCH},BATCH_SIZE=${BATCH_SIZE}" \
            slurm-scripts/run_3step_grid_search.sh | awk '{print $NF}')

    echo "Batch $BATCH/$((NUM_BATCHES-1))  OFFSET=$BATCH  array=0-${MAX}  jobid=$JOBID"

    # Wait for this batch to finish before submitting next
    if (( BATCH < NUM_BATCHES - 1 )); then
        echo "Waiting for job $JOBID to complete..."
        while squeue -j "$JOBID" -h 2>/dev/null | grep -q "$JOBID"; do
            sleep "$POLL_INTERVAL"
        done
        echo "Job $JOBID finished."
    fi
done
echo "All batches submitted."
