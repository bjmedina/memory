#!/usr/bin/env python3
"""Submit 3-step grid search in batches, waiting for each to finish.

Avoids QOSMaxSubmitJobPerUserLimit by polling squeue and only submitting
the next batch after the current one completes.

Usage:
    python slurm-scripts/submit_3step_batches.py
    python slurm-scripts/submit_3step_batches.py --dry-run
    python slurm-scripts/submit_3step_batches.py --num-batches 5
"""

import argparse
import subprocess
import sys
import time

SCRIPT = "slurm-scripts/run_3step_grid_search.sh"


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
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--num-batches", type=int, default=23,
                        help="Number of batches to submit (default: 23)")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Seconds between squeue checks (default: 30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sbatch commands without submitting")
    args = parser.parse_args()

    print(f"Num batches: {args.num_batches}")
    print(f"Array size:  0-149 (always)")
    print()

    for batch in range(args.num_batches):
        cmd = [
            "sbatch",
            "--array=0-149",
            f"--export=ALL,OFFSET={batch}",
            SCRIPT,
        ]
        stdout = run_cmd(cmd, dry_run=args.dry_run)
        job_id = stdout.split()[-1]

        print(f"[Batch {batch}/{args.num_batches - 1}]  OFFSET={batch}  jobid={job_id}")

        if batch < args.num_batches - 1 and not args.dry_run:
            print(f"  Waiting for job {job_id} to complete...")
            while job_is_running(job_id):
                time.sleep(args.poll_interval)
            print(f"  Job {job_id} finished.")

    print("\nAll batches submitted.")


if __name__ == "__main__":
    main()
