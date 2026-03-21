#!/bin/bash
# Submits 3-step grid search in batches, waiting for each to finish before
# submitting the next. This avoids QOSMaxSubmitJobPerUserLimit errors.
#
# Usage: bash slurm-scripts/submit_3step_batches.sh

NUM_BATCHES=23
POLL_INTERVAL=60  # seconds between squeue checks

for (( BATCH=0; BATCH<NUM_BATCHES; BATCH++ )); do
    JOBID=$(sbatch --array=0-149 \
            --export="ALL,OFFSET=${BATCH}" \
            slurm-scripts/run_3step_grid_search.sh | awk '{print $NF}')

    echo "Batch $BATCH/$((NUM_BATCHES-1))  OFFSET=$BATCH  jobid=$JOBID"

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
