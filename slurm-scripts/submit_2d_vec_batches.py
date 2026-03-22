#!/usr/bin/env python3
"""Submit 2D vectorized grid search in batches, waiting for each to finish.

Avoids QOSMaxSubmitJobPerUserLimit by polling squeue and only submitting
the next batch after the current one completes.

By default uses the fine grid (15 x 13 x 13 = 2535 jobs) with a batch size
of 150, yielding 17 batches.  The last batch is automatically shortened.

Usage:
    python slurm-scripts/submit_2d_vec_batches.py
    python slurm-scripts/submit_2d_vec_batches.py --dry-run
    python slurm-scripts/submit_2d_vec_batches.py --total-jobs 392 --batch-size 100
    python slurm-scripts/submit_2d_vec_batches.py --gather   # also submit gather job at end
"""

import argparse
import math
import subprocess
import sys
import time

SCRIPT = "slurm-scripts/run_2d_grid_search_vectorized.sh"
GATHER_SCRIPT = "slurm-scripts/gather_2d_grid_search_vectorized.sh"


def run_cmd(cmd: list[str], dry_run: bool = False) -> str:
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
    try:
        out = subprocess.run(
            ["squeue", "-j", job_id, "-h"],
            capture_output=True, text=True, timeout=30,
        )
        return bool(out.stdout.strip())
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--total-jobs", type=int, default=2535,
                        help="Total number of job indices (default: 2535 = fine grid)")
    parser.add_argument("--batch-size", type=int, default=150,
                        help="Max array tasks per batch (default: 150)")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between squeue checks (default: 60)")
    parser.add_argument("--gather", action="store_true",
                        help="Submit gather/merge job after all batches complete")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sbatch commands without submitting")
    args = parser.parse_args()

    num_batches = math.ceil(args.total_jobs / args.batch_size)

    print(f"Total jobs:  {args.total_jobs}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Num batches: {num_batches}")
    print()

    last_job_id = None
    for batch in range(num_batches):
        start = batch * args.batch_size
        end = min(start + args.batch_size, args.total_jobs) - 1
        array_spec = f"--array={start}-{end}"

        cmd = ["sbatch", array_spec, SCRIPT]
        stdout = run_cmd(cmd, dry_run=args.dry_run)
        job_id = stdout.split()[-1]
        last_job_id = job_id

        print(f"[Batch {batch}/{num_batches - 1}]  array={start}-{end}  jobid={job_id}")

        if batch < num_batches - 1 and not args.dry_run:
            print(f"  Waiting for job {job_id} to complete...")
            while job_is_running(job_id):
                time.sleep(args.poll_interval)
            print(f"  Job {job_id} finished.")

    print("\nAll batches submitted.")

    if args.gather:
        print("\nSubmitting gather/merge job...")
        gather_cmd = ["sbatch"]
        if last_job_id and not args.dry_run:
            gather_cmd.append(f"--dependency=afterok:{last_job_id}")
        gather_cmd.append(GATHER_SCRIPT)
        stdout = run_cmd(gather_cmd, dry_run=args.dry_run)
        gather_id = stdout.split()[-1]
        print(f"Gather job submitted: {gather_id}")


if __name__ == "__main__":
    main()
