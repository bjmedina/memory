#!/usr/bin/env python3
"""Submit prior-guided grid search in batches, waiting for each to finish.

Avoids QOSMaxSubmitJobPerUserLimit by polling squeue and only submitting
the next batch after the current one completes.

Grid sizes (default 15 x 15 x 15 = 3,375 configs):
  flat mode:   3,375 total jobs → ceil(3375/150) = 23 batches of 150
  sigma0 mode: 15 jobs total → 1 batch

Usage:
    python slurm-scripts/submit_prior_guided_batches.py
    python slurm-scripts/submit_prior_guided_batches.py --dry-run
    python slurm-scripts/submit_prior_guided_batches.py --num-batches 5
    python slurm-scripts/submit_prior_guided_batches.py --parallel-mode sigma0

    # Override grid search parameters:
    python slurm-scripts/submit_prior_guided_batches.py --metric euclidean --n-mc 5 --which-task 2

    # Auto-merge after all batches complete:
    python slurm-scripts/submit_prior_guided_batches.py --gather
"""

import argparse
import math
import subprocess
import sys
import time

SCRIPT = "slurm-scripts/run_prior_guided_grid_search.sh"
PYTHON_SCRIPT = "src/model/run_prior_guided_grid_search.py"
BATCH_SIZE = 150  # SLURM array tasks per batch


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


def compute_total_jobs(parallel_mode, n_sigma0=15, n_sigma=15, n_eta=15):
    if parallel_mode == 'sigma0':
        return n_sigma0
    return n_sigma0 * n_sigma * n_eta


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Batch control
    parser.add_argument("--num-batches", type=int, default=None,
                        help="Number of batches (auto-computed from grid if omitted)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"SLURM array tasks per batch (default: {BATCH_SIZE})")
    parser.add_argument("--poll-interval", type=int, default=30,
                        help="Seconds between squeue checks (default: 30)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print sbatch commands without submitting")
    parser.add_argument("--gather", action="store_true",
                        help="Run --merge after all batches complete")

    # Grid search parameters (passed through to SLURM script via env vars)
    parser.add_argument("--parallel-mode", type=str, default="flat",
                        choices=["flat", "sigma0"],
                        help="Parallelization strategy")
    parser.add_argument("--metric", type=str, default="cosine",
                        help="Distance metric")
    parser.add_argument("--n-mc", type=int, default=1,
                        help="Monte Carlo repetitions per config")
    parser.add_argument("--which-task", type=int, default=2,
                        help="Task index (0=env-sounds, 1=glob-music, 2=atexts)")
    parser.add_argument("--encoder", type=str, default="texture_pca",
                        help="Encoder type")
    parser.add_argument("--pc-dims", type=int, default=256,
                        help="Number of PC dimensions")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Override save directory (auto-generated if omitted)")

    # Grid sizes (for computing num_batches automatically)
    parser.add_argument("--n-sigma0", type=int, default=13,
                        help="Number of sigma0 grid values")
    parser.add_argument("--n-sigma", type=int, default=13,
                        help="Number of sigma grid values")
    parser.add_argument("--n-eta", type=int, default=13,
                        help="Number of eta grid values")

    args = parser.parse_args()

    # Auto-compute save dir
    if args.save_dir is None:
        args.save_dir = (
            f"/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/"
            f"reports/figures/prior_guided_grid_search_"
            f"metric-{args.metric}_nmc{args.n_mc}_task{args.which_task}"
        )

    # Auto-compute number of batches
    total_jobs = compute_total_jobs(
        args.parallel_mode, args.n_sigma0, args.n_sigma, args.n_eta
    )
    if args.num_batches is None:
        args.num_batches = math.ceil(total_jobs / args.batch_size)

    # For the last batch, the array may be shorter
    last_batch_size = total_jobs - (args.num_batches - 1) * args.batch_size

    print(f"=== Prior-guided grid search submission ===")
    print(f"  Total jobs:     {total_jobs}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Num batches:    {args.num_batches}")
    print(f"  Last batch:     {last_batch_size} jobs")
    print(f"  Parallel mode:  {args.parallel_mode}")
    print(f"  Metric:         {args.metric}")
    print(f"  N_MC:           {args.n_mc}")
    print(f"  Task:           {args.which_task}")
    print(f"  Encoder:        {args.encoder} (PC={args.pc_dims})")
    print(f"  Save dir:       {args.save_dir}")
    print()

    for batch in range(args.num_batches):
        # Compute array range for this batch
        start = 0
        end = args.batch_size - 1
        if batch == args.num_batches - 1 and last_batch_size < args.batch_size:
            end = last_batch_size - 1

        # Build environment exports
        exports = (
            f"ALL,"
            f"OFFSET={batch},"
            f"BATCH_SIZE={args.batch_size},"
            f"PARALLEL_MODE={args.parallel_mode},"
            f"METRIC={args.metric},"
            f"N_MC={args.n_mc},"
            f"WHICH_TASK={args.which_task},"
            f"ENCODER={args.encoder},"
            f"PC_DIMS={args.pc_dims},"
            f"SAVE_DIR={args.save_dir}"
        )

        cmd = [
            "sbatch",
            f"--array=0-{end}",
            f"--export={exports}",
            SCRIPT,
        ]

        stdout = run_cmd(cmd, dry_run=args.dry_run)
        job_id = stdout.split()[-1]

        print(f"[Batch {batch}/{args.num_batches - 1}]  "
              f"OFFSET={batch}  array=0-{end}  jobid={job_id}")

        if batch < args.num_batches - 1 and not args.dry_run:
            print(f"  Waiting for job {job_id} to complete...")
            while job_is_running(job_id):
                time.sleep(args.poll_interval)
            print(f"  Job {job_id} finished.")

    # Wait for the last batch too before gathering
    if args.gather and not args.dry_run:
        print(f"\nWaiting for final batch {job_id} to complete before merging...")
        while job_is_running(job_id):
            time.sleep(args.poll_interval)
        print(f"Final batch finished.")

    print("\nAll batches submitted.")

    # Auto-merge if requested
    if args.gather:
        print(f"\nMerging results from {args.save_dir} ...")
        merge_cmd = [
            "python", PYTHON_SCRIPT,
            "--merge",
            "--save-dir", args.save_dir,
        ]
        if args.dry_run:
            print(f"  [dry-run] {' '.join(merge_cmd)}")
        else:
            result = subprocess.run(merge_cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.returncode != 0:
                print(f"Merge failed: {result.stderr}", file=sys.stderr)
                sys.exit(1)
            print("Merge complete.")


if __name__ == "__main__":
    main()
