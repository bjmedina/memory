#!/usr/bin/env python3
"""
2D parameter grid search — standalone script for SLURM parallelization.

Sweeps (sigma0, sigma, eta) grid and computes d' per ISI using the
original (non-vectorised) runner ``run_model_core_2d``.

Parallelization modes
---------------------
  sigma0 (default):  Each job processes one sigma0 index → 8 jobs (default grid), 15 jobs (--fine).
  flat:              Each job processes one (sigma0, sigma, eta) triple → 392 (default) or 2535 (--fine).

Usage examples
--------------
  # Single slice locally
  python src/model/run_2d_grid_search.py --job-index 0 --n-mc 10

  # Merge all slices after completion
  python src/model/run_2d_grid_search.py --merge --save-dir reports/figures/2d_grid_search
"""

import sys
import os
import time
import argparse
from glob import glob
from collections import defaultdict

import numpy as np

# ── path setup ────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
sys.path.insert(0, _REPO_ROOT)

import torch
from src.model.analytic_gmm_2d import make_default_gmm
from src.model.score_adapter_2d import ScoreAdapter2D
from utls.sandbox_2d_data import make_2d_grid_stimuli
from utls.runners_2d import run_model_core_2d
from utls.toy_experiments import make_high_diversity_sequences
from utls.roc_utils import roc_from_arrays
from utls.analysis_helpers import auroc_to_dprime


# ── defaults ──────────────────────────────────────────────────────────
DEFAULT_SIGMA0 = [0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.8, 1.0]
DEFAULT_SIGMA  = [0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3]
DEFAULT_ETA    = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]
DEFAULT_ISIS   = [0, 2, 8, 16]

# Finer grid (~2× resolution per dimension) for --fine
FINE_SIGMA0 = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0]
FINE_SIGMA  = [0.0, 0.0125, 0.025, 0.0375, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3]
FINE_ETA    = [0.0, 0.0025, 0.005, 0.0075, 0.01, 0.015, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2]


# ── MC d-prime ────────────────────────────────────────────────────────

def run_mc_dprime(sigma0, sigma, eta, *,
                  X0, name_to_idx, experiment_list, adapter,
                  isi_values, n_mc, seed, metric):
    """Run MC sweep using original runner and return d' per ISI."""
    runner_isi_values = [isi + 1 for isi in isi_values]
    score_type = 'distance'

    all_isi_hits = defaultdict(list)
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
            all_isi_hits[risi].extend(run['isi_hit_dists'].get(risi, []))
        all_fas.extend(run['fas'])

    all_fas = np.array(all_fas, dtype=float)
    result = {}

    for exp_isi, risi in zip(isi_values, runner_isi_values):
        hits_raw = all_isi_hits.get(risi, [])
        if len(hits_raw) < 3:
            result[exp_isi] = np.nan
            continue
        hits = np.array([s for s, t in hits_raw], dtype=float)
        roc = roc_from_arrays(hits, all_fas, score_type=score_type)
        if roc is None:
            result[exp_isi] = np.nan
        else:
            _, _, auc_val = roc
            result[exp_isi] = auroc_to_dprime(auc_val)

    return result


# ── merge mode ────────────────────────────────────────────────────────

def merge_results(save_dir):
    """Merge per-slice .npz files into a single grid_search_results.npz."""
    # Try sigma0-mode files first
    sigma0_files = sorted(glob(os.path.join(save_dir, 'grid_slice_s0idx*.npz')))
    flat_files = sorted(glob(os.path.join(save_dir, 'grid_point_*.npz')))

    if sigma0_files:
        print(f'Found {len(sigma0_files)} sigma0-mode slice files')
        ref = np.load(sigma0_files[0])
        sigma0_grid = ref['sigma0_grid']
        sigma_grid = ref['sigma_grid']
        eta_grid = ref['eta_grid']
        isi_values = ref['isi_values']

        results = {int(isi): np.full((len(sigma0_grid), len(sigma_grid), len(eta_grid)), np.nan)
                   for isi in isi_values}

        for fpath in sigma0_files:
            data = np.load(fpath)
            i_s0 = int(data['sigma0_idx'])
            for isi in isi_values:
                key = f'dprime_isi{isi}'
                if key in data:
                    results[int(isi)][i_s0] = data[key]
            print(f'  Loaded slice s0_idx={i_s0} from {os.path.basename(fpath)}')

    elif flat_files:
        print(f'Found {len(flat_files)} flat-mode point files')
        ref = np.load(flat_files[0])
        sigma0_grid = ref['sigma0_grid']
        sigma_grid = ref['sigma_grid']
        eta_grid = ref['eta_grid']
        isi_values = ref['isi_values']

        results = {int(isi): np.full((len(sigma0_grid), len(sigma_grid), len(eta_grid)), np.nan)
                   for isi in isi_values}

        for fpath in flat_files:
            data = np.load(fpath)
            i_s0 = int(data['sigma0_idx'])
            i_sig = int(data['sigma_idx'])
            i_eta = int(data['eta_idx'])
            for isi in isi_values:
                key = f'dprime_isi{isi}'
                if key in data:
                    results[int(isi)][i_s0, i_sig, i_eta] = float(data[key])
        print(f'  Loaded {len(flat_files)} point files')

    else:
        print(f'No slice or point files found in {save_dir}')
        return

    # Check completeness
    total_expected = len(sigma0_grid) * len(sigma_grid) * len(eta_grid)
    any_isi = list(results.keys())[0]
    n_filled = int(np.sum(~np.isnan(results[any_isi])))
    print(f'\nFilled {n_filled}/{total_expected} grid points')
    if n_filled < total_expected:
        n_missing = total_expected - n_filled
        print(f'  WARNING: {n_missing} grid points are missing (NaN)')

    out_path = os.path.join(save_dir, 'grid_search_results.npz')
    np.savez(out_path,
             sigma0_grid=sigma0_grid,
             sigma_grid=sigma_grid,
             eta_grid=eta_grid,
             isi_values=isi_values,
             **{f'dprime_isi{isi}': results[int(isi)] for isi in isi_values})
    print(f'Saved merged results to {out_path}')


# ── main ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='2D parameter grid search with original (non-vectorised) runner',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Merge mode
    p.add_argument('--merge', action='store_true',
                   help='Merge per-slice .npz files instead of running')

    # Job control
    p.add_argument('--job-index', type=int, default=0,
                   help='SLURM_ARRAY_TASK_ID (0-based)')
    p.add_argument('--parallel-mode', type=str, default='sigma0',
                   choices=['sigma0', 'flat'],
                   help='Parallelization strategy')

    # Grid parameters
    p.add_argument('--fine', action='store_true',
                   help='Use finer grid (~2× resolution per dimension; more triples)')
    p.add_argument('--sigma0-grid', type=float, nargs='+', default=None,
                   help='Sigma0 (encoding noise) grid values (default: from --fine or coarse grid)')
    p.add_argument('--sigma-grid', type=float, nargs='+', default=None,
                   help='Sigma (diffusive noise) grid values')
    p.add_argument('--eta-grid', type=float, nargs='+', default=None,
                   help='Eta (drift step size) grid values')

    # Experiment parameters
    p.add_argument('--n-mc', type=int, default=10,
                   help='Monte Carlo repetitions per config')
    p.add_argument('--isis', type=int, nargs='+', default=DEFAULT_ISIS,
                   help='ISI values to evaluate')
    p.add_argument('--n-sequences', type=int, default=10,
                   help='Number of experiment sequences')
    p.add_argument('--seq-length', type=int, default=81,
                   help='Length of each sequence')
    p.add_argument('--min-pairs-per-isi', type=int, default=5,
                   help='Minimum repeat pairs per ISI per sequence')
    p.add_argument('--seed', type=int, default=42,
                   help='Base random seed')
    p.add_argument('--metric', type=str, default='euclidean',
                   help='Distance metric')

    # Output
    p.add_argument('--save-dir', type=str,
                   default='reports/figures/2d_grid_search',
                   help='Output directory for results')

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # ── merge mode ────────────────────────────────────────────────────
    if args.merge:
        merge_results(args.save_dir)
        return

    # ── grids ─────────────────────────────────────────────────────────
    if args.sigma0_grid is not None:
        sigma0_grid = np.array(args.sigma0_grid)
    else:
        sigma0_grid = np.array(FINE_SIGMA0 if args.fine else DEFAULT_SIGMA0)
    if args.sigma_grid is not None:
        sigma_grid = np.array(args.sigma_grid)
    else:
        sigma_grid = np.array(FINE_SIGMA if args.fine else DEFAULT_SIGMA)
    if args.eta_grid is not None:
        eta_grid = np.array(args.eta_grid)
    else:
        eta_grid = np.array(FINE_ETA if args.fine else DEFAULT_ETA)
    isi_values = tuple(args.isis)

    # ── validate job index ────────────────────────────────────────────
    if args.parallel_mode == 'sigma0':
        total_jobs = len(sigma0_grid)
    else:
        total_jobs = len(sigma0_grid) * len(sigma_grid) * len(eta_grid)

    if args.job_index >= total_jobs:
        print(f'ERROR: job-index {args.job_index} >= total jobs {total_jobs}')
        sys.exit(1)

    # ── setup (shared) ────────────────────────────────────────────────
    print(f'Setting up GMM, stimuli, and sequences ...')
    gmm = make_default_gmm()
    X0, name_to_idx, stimulus_pool = make_2d_grid_stimuli()
    adapter = ScoreAdapter2D(gmm, normalize=True)

    experiment_list, isi_keys = make_high_diversity_sequences(
        stimulus_pool=stimulus_pool,
        isi_values=list(isi_values),
        n_sequences=args.n_sequences,
        length=args.seq_length,
        min_pairs_per_isi=args.min_pairs_per_isi,
        seed=args.seed,
    )

    print(f'{len(experiment_list)} sequences, length {len(experiment_list[0])}')
    print(f'ISI values: {isi_values}')
    print(f'N_MC: {args.n_mc}')
    print(f'Metric: {args.metric}')
    print(f'Grid: {len(sigma0_grid)} x {len(sigma_grid)} x {len(eta_grid)} '
          f'= {len(sigma0_grid) * len(sigma_grid) * len(eta_grid)} configs')
    print(f'Parallel mode: {args.parallel_mode}, job index: {args.job_index}')
    print()

    common_kwargs = dict(
        X0=X0, name_to_idx=name_to_idx,
        experiment_list=experiment_list, adapter=adapter,
        isi_values=isi_values, n_mc=args.n_mc,
        seed=args.seed, metric=args.metric,
    )

    # ── dispatch ──────────────────────────────────────────────────────
    if args.parallel_mode == 'sigma0':
        _run_sigma0_slice(args, sigma0_grid, sigma_grid, eta_grid,
                          isi_values, common_kwargs)
    else:
        _run_flat_point(args, sigma0_grid, sigma_grid, eta_grid,
                        isi_values, common_kwargs)


def _run_sigma0_slice(args, sigma0_grid, sigma_grid, eta_grid,
                      isi_values, common_kwargs):
    """Process all (sigma, eta) combos for one sigma0 index."""
    i_s0 = args.job_index
    s0 = sigma0_grid[i_s0]

    n_configs = len(sigma_grid) * len(eta_grid)
    print(f'=== sigma0 slice: idx={i_s0}, sigma0={s0:.4f}, '
          f'{n_configs} configs ===')

    results = {isi: np.full((len(sigma_grid), len(eta_grid)), np.nan)
               for isi in isi_values}

    count = 0
    t_start = time.perf_counter()

    for i_sig, sig in enumerate(sigma_grid):
        for i_eta, eta in enumerate(eta_grid):
            dp = run_mc_dprime(s0, sig, eta, **common_kwargs)
            for isi in isi_values:
                results[isi][i_sig, i_eta] = dp.get(isi, np.nan)
            count += 1
            if count % 10 == 0 or count == n_configs:
                elapsed = time.perf_counter() - t_start
                rate = count / elapsed
                remaining = (n_configs - count) / rate if rate > 0 else 0
                print(f'  {count}/{n_configs} done  '
                      f'({elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining)')

    t_total = time.perf_counter() - t_start
    print(f'Slice complete: {count} configs in {t_total:.1f}s')

    out_path = os.path.join(args.save_dir, f'grid_slice_s0idx{i_s0}.npz')
    np.savez(out_path,
             sigma0_grid=sigma0_grid,
             sigma_grid=sigma_grid,
             eta_grid=eta_grid,
             isi_values=np.array(isi_values),
             sigma0_idx=i_s0,
             parallel_mode='sigma0',
             **{f'dprime_isi{isi}': results[isi] for isi in isi_values})
    print(f'Saved to {out_path}')

def _run_flat_point(args, sigma0_grid, sigma_grid, eta_grid,
                    isi_values, common_kwargs):
    """Process a single (sigma0, sigma, eta) config."""
    shape = (len(sigma0_grid), len(sigma_grid), len(eta_grid))
    i_s0, i_sig, i_eta = np.unravel_index(args.job_index, shape)

    s0 = sigma0_grid[i_s0]
    sig = sigma_grid[i_sig]
    eta = eta_grid[i_eta]

    print(f'=== flat point: idx={args.job_index}, '
          f'sigma0={s0:.4f}, sigma={sig:.4f}, eta={eta:.4f} ===')

    t_start = time.perf_counter()
    dp = run_mc_dprime(s0, sig, eta, **common_kwargs)
    t_total = time.perf_counter() - t_start

    print(f'Done in {t_total:.2f}s')
    for isi in isi_values:
        print(f"  ISI={isi}: d'={dp.get(isi, np.nan):.4f}")

    out_path = os.path.join(args.save_dir, f'grid_point_{args.job_index}.npz')
    np.savez(out_path,
             sigma0_grid=sigma0_grid,
             sigma_grid=sigma_grid,
             eta_grid=eta_grid,
             isi_values=np.array(isi_values),
             sigma0_idx=i_s0,
             sigma_idx=i_sig,
             eta_idx=i_eta,
             parallel_mode='flat',
             **{f'dprime_isi{isi}': np.float64(dp.get(isi, np.nan))
                for isi in isi_values})
    print(f'Saved to {out_path}')


if __name__ == '__main__':
    main()
