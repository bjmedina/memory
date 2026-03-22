#!/usr/bin/env python3
"""Submit 2D vectorized grid search in batches, waiting for each to finish.

Avoids QOSMaxSubmitJobPerUserLimit by polling squeue and only submitting
the next batch after the current one completes.

Grid presets (use --grid to select):
    default   8 × 7 × 7   =    392 jobs
    fine     15 × 13 × 13  =  2,535 jobs
    dense    30 × 25 × 26  = 19,500 jobs

Usage:
    python slurm-scripts/submit_2d_vec_batches.py --grid dense
    python slurm-scripts/submit_2d_vec_batches.py --grid dense --gather
    python slurm-scripts/submit_2d_vec_batches.py --grid dense --dry-run
    python slurm-scripts/submit_2d_vec_batches.py --total-jobs 5000   # custom count
"""

import argparse
import math
import os
import subprocess
import sys
import time

SCRIPT = "slurm-scripts/run_2d_grid_search_vectorized.sh"
GATHER_SCRIPT = "slurm-scripts/gather_2d_grid_search_vectorized.sh"

# Preset total-job counts (must match grids in run_2d_grid_search_vectorized.py)
GRID_PRESETS = {
    "default": {"total": 392,   "env": {}},
    "fine":    {"total": 2535,  "env": {"FINE_GRID": "true", "DENSE_GRID": "false"}},
    "dense":   {"total": 19500, "env": {"DENSE_GRID": "true", "FINE_GRID": "false"}},
}


def run_cmd(cmd: list[str], dry_run: bool = False, env: dict | None = None) -> str:
    if dry_run:
        env_str = " ".join(f"{k}={v}" for k, v in (env or {}).items())
        prefix = f"{env_str} " if env_str else ""
        print(f"  [dry-run] {prefix}{' '.join(cmd)}")
        return "Submitted batch job 00000"
    merged_env = {**os.environ, **(env or {})}
    result = subprocess.run(cmd, capture_output=True, text=True, env=merged_env)
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
    parser.add_argument("--grid", choices=list(GRID_PRESETS), default=None,
                        help="Grid preset (sets --total-jobs and env vars automatically)")
    parser.add_argument("--total-jobs", type=int, default=None,
                        help="Total number of job indices (overrides --grid count)")
    parser.add_argument("--batch-size", type=int, default=150,
                        help="Max array tasks per batch (default: 150)")
    parser.add_argument("--poll-interval", type=int, default=60,
                        help="Seconds between squeue checks (default: 60)")
    parser.add_argument("--gather", action="store_true",
                        help="Submit gather/merge job after all batches complete")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sbatch commands without submitting")
    args = parser.parse_args()

    # Resolve grid preset
    preset = GRID_PRESETS.get(args.grid or "fine")
    total_jobs = args.total_jobs if args.total_jobs is not None else preset["total"]
    extra_env = preset["env"] if args.grid else {}

    num_batches = math.ceil(total_jobs / args.batch_size)

    print(f"Grid:        {args.grid or 'fine'}")
    print(f"Total jobs:  {total_jobs}")
    print(f"Batch size:  {args.batch_size}")
    print(f"Num batches: {num_batches}")
    if extra_env:
        print(f"Env vars:    {extra_env}")
    print()

    last_job_id = None
    for batch in range(num_batches):
        start = batch * args.batch_size
        end = min(start + args.batch_size, total_jobs) - 1
        array_spec = f"--array={start}-{end}"

        cmd = ["sbatch", array_spec, SCRIPT]
        stdout = run_cmd(cmd, dry_run=args.dry_run, env=extra_env)
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
        stdout = run_cmd(gather_cmd, dry_run=args.dry_run, env=extra_env)
        gather_id = stdout.split()[-1]
        print(f"Gather job submitted: {gather_id}")


if __name__ == "__main__":
    main()
