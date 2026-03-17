#!/usr/bin/env python3
"""
Standalone runner for the 2D parameter grid search.

Sweeps (σ₀, σ, η) and saves **comprehensive** per-triple data so that
any configuration can be queried after the fact:

    python scripts/run_2d_grid_search.py --output-dir results/my_run
    python scripts/run_2d_grid_search.py --resume   # skip already-computed triples

SLURM job-array mode (one triple per task):

    python scripts/run_2d_grid_search.py --output-dir results/run \
        --task-id $SLURM_ARRAY_TASK_ID --num-tasks 392

Gather results after all array jobs finish:

    python scripts/run_2d_grid_search.py --output-dir results/run --gather

Output structure
----------------
<output-dir>/
    grid_search_master.csv          # one row per triple, all summary metrics
    grid_search_results.npz         # 3-D d' arrays (backward-compat w/ notebook)
    per_triple/                     # one .npz per (σ₀, σ, η) with raw scores
        s0=0.000_sig=0.000_eta=0.000.npz
        ...

To look up a specific triple after the run:

    >>> import pandas as pd, numpy as np
    >>> df = pd.read_csv('grid_search_master.csv')
    >>> row = df[(df.sigma0 == 0.5) & (df.sigma == 0.1) & (df.eta == 0.02)]
    >>> data = np.load('per_triple/s0=0.500_sig=0.100_eta=0.020.npz',
    ...               allow_pickle=True)
    >>> data['hit_scores_isi0']   # raw cosine-distance scores for hits
    >>> data['roc_fpr_isi0']      # ROC curve false-positive rates
"""

import sys, os, argparse, time, csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch

# ── project imports (run from repo root) ─────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model.analytic_gmm_2d import make_default_gmm
from src.model.score_adapter_2d import ScoreAdapter2D
from utls.sandbox_2d_data import make_2d_grid_stimuli
from utls.runners_2d import run_model_core_2d
from utls.toy_experiments import make_high_diversity_sequences
from utls.roc_utils import roc_from_arrays
from utls.analysis_helpers import auroc_to_dprime, bootstrap_dprime_ci


# ── default grids (same as notebook) ─────────────────────────────────
DEFAULT_SIGMA0 = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
DEFAULT_SIGMA  = [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3]
DEFAULT_ETA    = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]


def triple_filename(s0, sig, eta):
    return f"s0={s0:.3f}_sig={sig:.3f}_eta={eta:.3f}.npz"


def run_single_triple(
    sigma0, sigma, eta,
    *,
    X0, name_to_idx, experiment_list, adapter,
    isi_values, n_mc, seed, metric,
):
    """Run MC reps for one (σ₀, σ, η) and return comprehensive results."""
    runner_isi_values = [isi + 1 for isi in isi_values]
    score_type = "distance"

    # Aggregate across MC reps
    all_isi_hits = defaultdict(list)       # runner_isi → [(score, t), ...]
    all_fas = []

    for rep in range(n_mc):
        run = run_model_core_2d(
            sigma0=sigma0, sigma=sigma,
            X0=X0, name_to_idx=name_to_idx,
            experiment_list=experiment_list,
            score_model=adapter,
            drift_step_size=eta,
            metric=metric,
            seed=seed * 10_000 + rep,
        )
        for risi in runner_isi_values:
            all_isi_hits[risi].extend(run["isi_hit_dists"].get(risi, []))
        all_fas.extend(run["fas"])

    fas_arr = np.asarray(all_fas, dtype=float)

    # Per-ISI analysis
    triple_data = {
        "sigma0": sigma0,
        "sigma": sigma,
        "eta": eta,
        "n_mc": n_mc,
        "seed": seed,
        "metric": metric,
        "isi_values": np.array(list(isi_values)),
        "fa_scores": fas_arr,
        "fa_mean": float(np.mean(fas_arr)) if len(fas_arr) > 0 else np.nan,
        "fa_std": float(np.std(fas_arr)) if len(fas_arr) > 0 else np.nan,
        "n_fas": len(fas_arr),
    }

    summary = {
        "sigma0": sigma0,
        "sigma": sigma,
        "eta": eta,
        "n_fas": len(fas_arr),
        "fa_mean": triple_data["fa_mean"],
        "fa_std": triple_data["fa_std"],
    }

    for exp_isi, risi in zip(isi_values, runner_isi_values):
        hits_raw = all_isi_hits.get(risi, [])
        n_hits = len(hits_raw)

        if n_hits < 3:
            # Not enough data
            triple_data[f"hit_scores_isi{exp_isi}"] = np.array([], dtype=float)
            triple_data[f"hit_timestamps_isi{exp_isi}"] = np.array([], dtype=int)
            triple_data[f"roc_fpr_isi{exp_isi}"] = np.array([], dtype=float)
            triple_data[f"roc_tpr_isi{exp_isi}"] = np.array([], dtype=float)
            for key in ["dprime", "auc", "dprime_sem", "n_hits", "hit_mean", "hit_std"]:
                k = f"{key}_isi{exp_isi}"
                triple_data[k] = np.nan
                summary[k] = np.nan
            continue

        hits_scores = np.array([s for s, t in hits_raw], dtype=float)
        hits_times = np.array([t for s, t in hits_raw], dtype=int)

        # ROC + AUC
        roc = roc_from_arrays(hits_scores, fas_arr, score_type=score_type)
        if roc is not None:
            fpr, tpr, auc_val = roc
            dp = auroc_to_dprime(auc_val)
        else:
            fpr, tpr = np.array([]), np.array([])
            auc_val, dp = np.nan, np.nan

        # Bootstrap CI
        run_data = {
            "isi_hit_dists": {risi: [(h, 0) for h in hits_scores]},
            "fas": fas_arr,
            "score_type": score_type,
        }
        mean_dp, sem_dp = bootstrap_dprime_ci(run_data, risi, n_boot=200)
        if not np.isfinite(mean_dp):
            mean_dp = dp
        if not np.isfinite(sem_dp):
            sem_dp = 0.0

        # Store in triple data
        triple_data[f"hit_scores_isi{exp_isi}"] = hits_scores
        triple_data[f"hit_timestamps_isi{exp_isi}"] = hits_times
        triple_data[f"roc_fpr_isi{exp_isi}"] = fpr
        triple_data[f"roc_tpr_isi{exp_isi}"] = tpr
        triple_data[f"auc_isi{exp_isi}"] = float(auc_val)
        triple_data[f"dprime_isi{exp_isi}"] = float(mean_dp)
        triple_data[f"dprime_sem_isi{exp_isi}"] = float(sem_dp)
        triple_data[f"n_hits_isi{exp_isi}"] = n_hits
        triple_data[f"hit_mean_isi{exp_isi}"] = float(np.mean(hits_scores))
        triple_data[f"hit_std_isi{exp_isi}"] = float(np.std(hits_scores))

        # Summary row
        summary[f"dprime_isi{exp_isi}"] = float(mean_dp)
        summary[f"auc_isi{exp_isi}"] = float(auc_val)
        summary[f"dprime_sem_isi{exp_isi}"] = float(sem_dp)
        summary[f"n_hits_isi{exp_isi}"] = n_hits
        summary[f"hit_mean_isi{exp_isi}"] = float(np.mean(hits_scores))
        summary[f"hit_std_isi{exp_isi}"] = float(np.std(hits_scores))

    return triple_data, summary


def make_csv_fields(isi_values):
    """Build ordered CSV column names."""
    fields = ["sigma0", "sigma", "eta", "n_fas", "fa_mean", "fa_std"]
    for isi in isi_values:
        fields.extend([
            f"dprime_isi{isi}", f"auc_isi{isi}", f"dprime_sem_isi{isi}",
            f"n_hits_isi{isi}", f"hit_mean_isi{isi}", f"hit_std_isi{isi}",
        ])
    return fields


def gather_results(out_dir, per_triple_dir, sigma0_grid, sigma_grid,
                   eta_grid, isi_values):
    """Assemble per_triple/*.npz into master CSV and grid .npz."""
    import re

    csv_fields = make_csv_fields(isi_values)
    csv_path = out_dir / "grid_search_master.csv"

    # Build index maps for grid coordinates
    s0_idx = {round(float(v), 6): i for i, v in enumerate(sigma0_grid)}
    sig_idx = {round(float(v), 6): i for i, v in enumerate(sigma_grid)}
    eta_idx = {round(float(v), 6): i for i, v in enumerate(eta_grid)}

    dprime_arrays = {
        isi: np.full((len(sigma0_grid), len(sigma_grid), len(eta_grid)), np.nan)
        for isi in isi_values
    }

    npz_files = sorted(per_triple_dir.glob("s0=*.npz"))
    print(f"Gathering {len(npz_files)} per-triple .npz files from {per_triple_dir}")

    rows = []
    for fpath in npz_files:
        data = np.load(fpath, allow_pickle=True)
        s0 = float(data["sigma0"])
        sig = float(data["sigma"])
        eta = float(data["eta"])

        row = {
            "sigma0": s0, "sigma": sig, "eta": eta,
            "n_fas": int(data["n_fas"]),
            "fa_mean": float(data["fa_mean"]),
            "fa_std": float(data["fa_std"]),
        }
        for isi in isi_values:
            for key in ["dprime", "auc", "dprime_sem", "n_hits", "hit_mean", "hit_std"]:
                k = f"{key}_isi{isi}"
                row[k] = float(data[k]) if k in data else np.nan

        rows.append(row)

        # Fill 3-D arrays
        i_s0 = s0_idx.get(round(s0, 6))
        i_sig = sig_idx.get(round(sig, 6))
        i_eta = eta_idx.get(round(eta, 6))
        if i_s0 is not None and i_sig is not None and i_eta is not None:
            for isi in isi_values:
                k = f"dprime_isi{isi}"
                if k in data:
                    dprime_arrays[isi][i_s0, i_sig, i_eta] = float(data[k])

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    # Write grid .npz
    npz_path = out_dir / "grid_search_results.npz"
    np.savez(
        npz_path,
        sigma0_grid=sigma0_grid,
        sigma_grid=sigma_grid,
        eta_grid=eta_grid,
        isi_values=np.array(isi_values),
        **{f"dprime_isi{isi}": dprime_arrays[isi] for isi in isi_values},
    )

    total = len(sigma0_grid) * len(sigma_grid) * len(eta_grid)
    print(f"  CSV:       {csv_path}  ({len(rows)} rows)")
    print(f"  Grid .npz: {npz_path}")
    if len(rows) < total:
        print(f"  WARNING: only {len(rows)}/{total} triples found")


def main():
    parser = argparse.ArgumentParser(
        description="Run 2D parameter grid search with comprehensive output."
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="reports/figures/2d_grid_search",
        help="Directory for all output files.",
    )
    parser.add_argument("--n-mc", type=int, default=10, help="Monte Carlo reps per triple.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=200, help="Bootstrap resamples for SEM.")
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip triples whose per_triple/*.npz already exists.",
    )
    parser.add_argument(
        "--sigma0-grid", type=float, nargs="+", default=DEFAULT_SIGMA0,
        help="σ₀ values to sweep.",
    )
    parser.add_argument(
        "--sigma-grid", type=float, nargs="+", default=DEFAULT_SIGMA,
        help="σ values to sweep.",
    )
    parser.add_argument(
        "--eta-grid", type=float, nargs="+", default=DEFAULT_ETA,
        help="η values to sweep.",
    )
    # ── SLURM job-array support ──────────────────────────────────────
    parser.add_argument(
        "--task-id", type=int, default=None,
        help="Run only this triple index (0-based). For SLURM --array jobs.",
    )
    parser.add_argument(
        "--num-tasks", type=int, default=None,
        help="Expected total number of tasks (for validation).",
    )
    parser.add_argument(
        "--gather", action="store_true",
        help="Gather per_triple/*.npz into master CSV + grid .npz (no simulation).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    per_triple_dir = out_dir / "per_triple"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_triple_dir.mkdir(exist_ok=True)

    sigma0_grid = np.array(args.sigma0_grid)
    sigma_grid = np.array(args.sigma_grid)
    eta_grid = np.array(args.eta_grid)
    total = len(sigma0_grid) * len(sigma_grid) * len(eta_grid)

    ISI_VALUES = (0, 2, 16)
    METRIC = "cosine"

    # ── gather mode ──────────────────────────────────────────────────
    if args.gather:
        gather_results(out_dir, per_triple_dir, sigma0_grid, sigma_grid,
                       eta_grid, ISI_VALUES)
        return

    # ── build flat list of triples (deterministic order) ─────────────
    all_triples = [
        (i_s0, i_sig, i_eta, s0, sig, eta)
        for i_s0, s0 in enumerate(sigma0_grid)
        for i_sig, sig in enumerate(sigma_grid)
        for i_eta, eta in enumerate(eta_grid)
    ]
    assert len(all_triples) == total

    # ── single-task mode (SLURM job array) ───────────────────────────
    if args.task_id is not None:
        if args.num_tasks is not None and args.num_tasks != total:
            print(f"WARNING: --num-tasks {args.num_tasks} != grid size {total}")
        if args.task_id < 0 or args.task_id >= total:
            print(f"ERROR: --task-id {args.task_id} out of range [0, {total})")
            sys.exit(1)

        i_s0, i_sig, i_eta, s0, sig, eta = all_triples[args.task_id]
        fname = triple_filename(s0, sig, eta)
        fpath = per_triple_dir / fname

        if args.resume and fpath.exists():
            print(f"Task {args.task_id}: {fname} already exists, skipping.")
            return

        print(f"Task {args.task_id}/{total}: s0={s0:.3f} sig={sig:.3f} eta={eta:.3f}")
        print(f"MC reps: {args.n_mc}  |  seed: {args.seed}")

        # Setup
        gmm = make_default_gmm()
        X0, name_to_idx, stimulus_pool = make_2d_grid_stimuli()
        adapter = ScoreAdapter2D(gmm, normalize=True)
        experiment_list, _ = make_high_diversity_sequences(
            stimulus_pool=stimulus_pool,
            isi_values=list(ISI_VALUES),
            n_sequences=10, length=81,
            min_pairs_per_isi=5, seed=args.seed,
        )

        t0 = time.time()
        triple_data, summary = run_single_triple(
            s0, sig, eta,
            X0=X0, name_to_idx=name_to_idx,
            experiment_list=experiment_list, adapter=adapter,
            isi_values=ISI_VALUES, n_mc=args.n_mc,
            seed=args.seed, metric=METRIC,
        )
        np.savez(fpath, **triple_data)

        elapsed = time.time() - t0
        print(
            f"  d'(0)={summary.get('dprime_isi0', float('nan')):.2f}  "
            f"d'(2)={summary.get('dprime_isi2', float('nan')):.2f}  "
            f"d'(16)={summary.get('dprime_isi16', float('nan')):.2f}  "
            f"[{elapsed:.1f}s]"
        )
        print(f"  Saved: {fpath}")
        return

    # ── sequential mode (original behavior) ──────────────────────────
    print(f"Grid: {len(sigma0_grid)} x {len(sigma_grid)} x {len(eta_grid)} = {total} triples")
    print(f"MC reps: {args.n_mc}  |  seed: {args.seed}  |  output: {out_dir}")
    if args.resume:
        print("Resume mode: skipping existing per_triple/*.npz files")
    print()

    # ── setup ────────────────────────────────────────────────────────
    gmm = make_default_gmm()
    X0, name_to_idx, stimulus_pool = make_2d_grid_stimuli()
    adapter = ScoreAdapter2D(gmm, normalize=True)

    experiment_list, _ = make_high_diversity_sequences(
        stimulus_pool=stimulus_pool,
        isi_values=list(ISI_VALUES),
        n_sequences=10,
        length=81,
        min_pairs_per_isi=5,
        seed=args.seed,
    )
    print(f"Generated {len(experiment_list)} sequences of length {len(experiment_list[0])}")

    # ── CSV setup ────────────────────────────────────────────────────
    csv_path = out_dir / "grid_search_master.csv"
    csv_fields = make_csv_fields(ISI_VALUES)

    # If resuming, load existing CSV rows to avoid duplicates
    existing_keys = set()
    if args.resume and csv_path.exists():
        import csv as csv_mod
        with open(csv_path, "r") as f:
            reader = csv_mod.DictReader(f)
            for row in reader:
                key = (float(row["sigma0"]), float(row["sigma"]), float(row["eta"]))
                existing_keys.add(key)
        # Reopen in append mode
        csv_file = open(csv_path, "a", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    else:
        csv_file = open(csv_path, "w", newline="")
        writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        writer.writeheader()

    # ── 3-D arrays for backward-compat .npz ──────────────────────────
    dprime_arrays = {
        isi: np.full((len(sigma0_grid), len(sigma_grid), len(eta_grid)), np.nan)
        for isi in ISI_VALUES
    }

    # ── main loop ────────────────────────────────────────────────────
    count = 0
    skipped = 0
    t_start = time.time()

    for i_s0, i_sig, i_eta, s0, sig, eta in all_triples:
        count += 1
        fname = triple_filename(s0, sig, eta)
        fpath = per_triple_dir / fname

        # Resume check
        if args.resume and fpath.exists():
            # Load d' from existing file into arrays
            try:
                existing = np.load(fpath, allow_pickle=True)
                for isi in ISI_VALUES:
                    key = f"dprime_isi{isi}"
                    if key in existing:
                        dprime_arrays[isi][i_s0, i_sig, i_eta] = float(existing[key])
            except Exception:
                pass
            skipped += 1
            continue

        # Run simulation
        triple_data, summary = run_single_triple(
            s0, sig, eta,
            X0=X0, name_to_idx=name_to_idx,
            experiment_list=experiment_list, adapter=adapter,
            isi_values=ISI_VALUES, n_mc=args.n_mc,
            seed=args.seed, metric=METRIC,
        )

        # Save per-triple .npz (crash-safe: write immediately)
        np.savez(fpath, **triple_data)

        # Write CSV row
        writer.writerow(summary)
        csv_file.flush()

        # Update 3-D arrays
        for isi in ISI_VALUES:
            dprime_arrays[isi][i_s0, i_sig, i_eta] = summary.get(
                f"dprime_isi{isi}", np.nan
            )

        # Progress
        done = count - skipped
        elapsed = time.time() - t_start
        rate = elapsed / max(done, 1)
        remaining = rate * (total - count)
        if done % 10 == 0 or done <= 3:
            print(
                f"  [{count}/{total}] "
                f"s0={s0:.3f} sig={sig:.3f} eta={eta:.3f}  "
                f"d'(0)={summary.get('dprime_isi0', float('nan')):.2f}  "
                f"d'(2)={summary.get('dprime_isi2', float('nan')):.2f}  "
                f"d'(16)={summary.get('dprime_isi16', float('nan')):.2f}  "
                f"[{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining]"
            )

    csv_file.close()

    # ── save backward-compat grid .npz ───────────────────────────────
    npz_path = out_dir / "grid_search_results.npz"
    np.savez(
        npz_path,
        sigma0_grid=sigma0_grid,
        sigma_grid=sigma_grid,
        eta_grid=eta_grid,
        isi_values=np.array(ISI_VALUES),
        **{f"dprime_isi{isi}": dprime_arrays[isi] for isi in ISI_VALUES},
    )

    elapsed = time.time() - t_start
    print()
    print(f"Done in {elapsed:.1f}s  ({count} total, {skipped} skipped)")
    print(f"  CSV:         {csv_path}")
    print(f"  Grid .npz:   {npz_path}")
    print(f"  Per-triple:  {per_triple_dir}/  ({count - skipped} files)")


if __name__ == "__main__":
    main()
