#!/usr/bin/env python3
"""Submit 3-step grid search in batches, waiting for each to finish.

Avoids QOSMaxSubmitJobPerUserLimit by polling squeue and only submitting
the next batch after the current one completes.

Usage:
    python slurm-scripts/submit_3step_batches.py
    python slurm-scripts/submit_3step_batches.py --dry-run
    python slurm-scripts/submit_3step_batches.py --total-jobs 100 --batch-size 50
"""

import argparse
import math
import subprocess
import sys
import time


def run_cmd(cmd: list[str], dry_run: bool = False) -> str:
    """Run a shell command and return stdout."""
    if dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return "Submitted batch job 00000"
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: {' '.join(cmd)}", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        sys.exit(1)
    return result.stdout.strip()


def job_is_running(job_id: str) -> bool:
    """Check if a SLURM job (or any of its array tasks) is still queued/running."""
    try:
        out = subprocess.run(
            ["squeue", "-j", job_id, "-h"],
            capture_output=True, text=True, timeout=30,
        )
        return bool(out.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def submit_batch(batch: int, max_index: int, batch_size: int,
                 script: str, dry_run: bool) -> str:
    """Submit one sbatch array and return the job ID."""
    cmd = [
        "sbatch",
        f"--array=0-{max_index}",
        f"--export=ALL,OFFSET={batch},BATCH_SIZE={batch_size}",
        script,
    ]
    stdout = run_cmd(cmd, dry_run=dry_run)
    # "Submitted batch job 12345678"
    job_id = stdout.split()[-1]
    return job_id


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--total-jobs", type=int, default=3375,
                        help="Total number of flat-index jobs (default: 3375)")
    parser.add_argument("--batch-size", type=int, default=150,
                        help="Max array jobs per batch (default: 150)")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between squeue checks (default: 60)")
    parser.add_argument("--script", default="slurm-scripts/run_3step_grid_search.sh",
                        help="Path to the SLURM job script")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sbatch commands without submitting")
    args = parser.parse_args()

    num_batches = math.ceil(args.total_jobs / args.batch_size)

    print(f"Total jobs:  {args.total_jobs}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Num batches: {num_batches}")
    print()

    for batch in range(num_batches):
        start = batch * args.batch_size
        remaining = args.total_jobs - start
        max_index = min(remaining, args.batch_size) - 1

        job_id = submit_batch(batch, max_index, args.batch_size,
                              args.script, args.dry_run)

        print(f"[Batch {batch}/{num_batches - 1}]  "
              f"OFFSET={batch}  array=0-{max_index}  "
              f"jobs {start}-{start + max_index}  jobid={job_id}")

        # Wait for batch to finish before submitting the next one
        if batch < num_batches - 1 and not args.dry_run:
            print(f"  Waiting for job {job_id} to complete...")
            while job_is_running(job_id):
                time.sleep(args.poll_interval)
            print(f"  Job {job_id} finished.")

    print("\nAll batches submitted.")


if __name__ == "__main__":
    main()
