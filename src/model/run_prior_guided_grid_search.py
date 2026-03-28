#!/usr/bin/env python3
"""
Prior-guided drift-diffusion grid search — standalone script for SLURM parallelization.

Sweeps (sigma0, sigma, eta) grid and computes d' per ISI using
``run_model_core_prior`` with the learned score function (ScoreFunction)
and real stimulus representations (256-PC auditory texture space).

This is the full-dimensional analogue of the 2D sandbox grid search
(``run_2d_grid_search_vectorized.py``), using the same model but with:
  - Real auditory texture stimuli (not 2D grid points)
  - Learned score function (not analytic GMM)
  - 256-dimensional PC space (not 2D)

Parallelization modes
---------------------
  flat (default):    Each job processes one (sigma0, sigma, eta) triple.
  sigma0:            Each job processes one sigma0 index.

Resume
------
  --resume       Skip triples that already have a per-triple .npz file.

Output is written under --save-dir.
Merged file: grid_search_results_prior_guided.npz.

Usage examples
--------------
  # Single triple locally
  python src/model/run_prior_guided_grid_search.py --job-index 0 --n-mc 10

  # Merge all slices after completion
  python src/model/run_prior_guided_grid_search.py --merge --save-dir reports/figures/prior_guided_grid_search
"""

import sys
import os
import time
import argparse
from glob import glob
from collections import defaultdict

import numpy as np

import types

# Mock 'cox' and 'cox.store' so constants.py doesn't crash
cox_mock = types.ModuleType('cox')
store_mock = types.ModuleType('cox.store')
store_mock.PYTORCH_STATE = 'pytorch_state'
cox_mock.store = store_mock
sys.modules['cox'] = cox_mock
sys.modules['cox.store'] = store_mock

# ── path setup ────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
sys.path.insert(0, _REPO_ROOT)

# cluster paths (needed for encoder pipeline + score model)
sys.path.append('/orcd/data/jhm/001/om2/jmhicks/projects/TextureStreaming/code/')
sys.path.append('/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/utls/')
sys.path.append('/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/src/model/')
sys.path.append('/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/')

import torch

from chexture_choolbox.auditorytexture.texture_model import TextureModel
from chexture_choolbox.auditorytexture.helpers import FlattenStats
from texture_prior.params import model_params, statistics_set

from utls.encoders import *
from utls.runners_prior import run_model_core_prior
from utls.runners_utils import load_experiment_data, build_encoder, encode_stimuli
from utls.toy_experiments import make_high_diversity_sequences
from utls.roc_utils import roc_from_arrays
from utls.analysis_helpers import auroc_to_dprime
from src.model.ScoreFunction import ScoreFunction


# ── defaults ──────────────────────────────────────────────────────────
# Grid values for (sigma0, sigma, eta)
DEFAULT_SIGMA0 = [0.0] + np.linspace(0.01, 1.0, 12).tolist()
# sigma: diffusive noise per step — moderate range for Langevin dynamics
DEFAULT_SIGMA  = [0.0] + np.linspace(0.01, 1.0, 12).tolist()
# eta: drift step size — typically small
DEFAULT_ETA    = [0.0] + np.linspace(0.01, 1.0, 12).tolist()

DEFAULT_ISIS   = [0, 1, 2, 4, 8, 16, 32, 64]

# ── MC d-prime ────────────────────────────────────────────────────────

def run_mc_dprime(sigma0, sigma, eta, *,
                  X0, name_to_idx, experiment_list, score_model,
                  isi_values, n_mc, seed, metric):
    """Run MC sweep using prior-guided runner and return d' per ISI.

    Returns
    -------
    dprime_dict : dict
        {isi: d'} for each ISI.
    triple_data : dict
        Comprehensive per-triple data including raw scores, ROC curves,
        and summary statistics (suitable for saving as .npz).
    """
    runner_isi_values = [isi + 1 for isi in isi_values]
    score_type = 'distance' if metric != 'loglikelihood' else 'likelihood'

    all_isi_hits = defaultdict(list)
    all_fas = []

    for rep in range(n_mc):
        run = run_model_core_prior(
            sigma0=sigma0, sigma=sigma,
            X0=X0, name_to_idx=name_to_idx,
            experiment_list=experiment_list,
            score_model=score_model,
            drift_step_size=eta,
            metric=metric,
            seed=seed * 10_000 + rep,
        )
        for risi in runner_isi_values:
            all_isi_hits[risi].extend(run['isi_hit_dists'].get(risi, []))
        all_fas.extend(run['fas'])

    fas_arr = np.array(all_fas, dtype=float)

    dprime_dict = {}
    triple_data = {
        'sigma0': sigma0,
        'sigma': sigma,
        'eta': eta,
        'n_mc': n_mc,
        'seed': seed,
        'metric': metric,
        'isi_values': np.array(list(isi_values)),
        'fa_scores': fas_arr,
        'fa_mean': float(np.mean(fas_arr)) if len(fas_arr) > 0 else np.nan,
        'fa_std': float(np.std(fas_arr)) if len(fas_arr) > 0 else np.nan,
        'n_fas': len(fas_arr),
    }

    for exp_isi, risi in zip(isi_values, runner_isi_values):
        hits_raw = all_isi_hits.get(risi, [])
        n_hits = len(hits_raw)

        if n_hits < 3:
            dprime_dict[exp_isi] = np.nan
            triple_data[f'hit_scores_isi{exp_isi}'] = np.array([], dtype=float)
            triple_data[f'hit_timestamps_isi{exp_isi}'] = np.array([], dtype=int)
            triple_data[f'roc_fpr_isi{exp_isi}'] = np.array([], dtype=float)
            triple_data[f'roc_tpr_isi{exp_isi}'] = np.array([], dtype=float)
            for key in ['dprime', 'auc', 'dprime_sem', 'n_hits', 'hit_mean', 'hit_std']:
                triple_data[f'{key}_isi{exp_isi}'] = np.nan
            continue

        hits_scores = np.array([s for s, t in hits_raw], dtype=float)
        hits_times = np.array([t for s, t in hits_raw], dtype=int)

        roc = roc_from_arrays(hits_scores, fas_arr, score_type=score_type)
        if roc is not None:
            fpr, tpr, auc_val = roc
            dp = auroc_to_dprime(auc_val)
        else:
            fpr, tpr = np.array([]), np.array([])
            auc_val, dp = np.nan, np.nan

        dprime_dict[exp_isi] = dp

        triple_data[f'hit_scores_isi{exp_isi}'] = hits_scores
        triple_data[f'hit_timestamps_isi{exp_isi}'] = hits_times
        triple_data[f'roc_fpr_isi{exp_isi}'] = fpr
        triple_data[f'roc_tpr_isi{exp_isi}'] = tpr
        triple_data[f'auc_isi{exp_isi}'] = float(auc_val)
        triple_data[f'dprime_isi{exp_isi}'] = float(dp)
        triple_data[f'dprime_sem_isi{exp_isi}'] = 0.0
        triple_data[f'n_hits_isi{exp_isi}'] = n_hits
        triple_data[f'hit_mean_isi{exp_isi}'] = float(np.mean(hits_scores))
        triple_data[f'hit_std_isi{exp_isi}'] = float(np.std(hits_scores))

    return dprime_dict, triple_data

# ── merge mode ────────────────────────────────────────────────────────

def merge_results(save_dir):
    """Merge per-slice .npz files into a single grid_search_results_prior_guided.npz."""
    sigma0_files = sorted(glob.glob(os.path.join(save_dir, 'grid_slice_s0idx*.npz')))
    flat_files = sorted(glob.glob(os.path.join(save_dir, 'grid_point_*.npz')))

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

    out_path = os.path.join(save_dir, 'grid_search_results_prior_guided.npz')
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
        description='Prior-guided drift-diffusion grid search (σ₀, σ, η) with learned score function',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Merge mode
    p.add_argument('--merge', action='store_true',
                   help='Merge per-slice .npz files instead of running')

    # Resume
    p.add_argument('--resume', action='store_true',
                   help='Skip triples that already have per-triple .npz files')

    # Job control
    p.add_argument('--job-index', type=int, default=0,
                   help='SLURM_ARRAY_TASK_ID (0-based)')
    p.add_argument('--parallel-mode', type=str, default='flat',
                   choices=['flat', 'sigma0'],
                   help='Parallelization strategy')

    # Grid parameters
    p.add_argument('--sigma0-grid', type=float, nargs='+', default=None,
                   help='Sigma0 (encoding noise) grid values')
    p.add_argument('--sigma-grid', type=float, nargs='+', default=None,
                   help='Sigma (diffusive noise) grid values')
    p.add_argument('--eta-grid', type=float, nargs='+', default=None,
                   help='Eta (drift step size) grid values')

    # Experiment parameters
    p.add_argument('--n-mc', type=int, default=1,
                   help='Monte Carlo repetitions per config')
    p.add_argument('--isis', type=int, nargs='+', default=DEFAULT_ISIS,
                   help='ISI values to evaluate')
    p.add_argument('--n-sequences', type=int, default=300,
                   help='Number of experiment sequences')
    p.add_argument('--seq-length', type=int, default=135,
                   help='Length of each sequence')
    p.add_argument('--min-pairs-per-isi', type=int, default=4,
                   help='Minimum repeat pairs per ISI per sequence')
    p.add_argument('--seed', type=int, default=44,
                   help='Base random seed')
    p.add_argument('--metric', type=str, default='cosine',
                   help='Distance metric')

    # Experiment data
    p.add_argument('--which-task', type=int, default=2,
                   help='Task index (0=env-sounds, 1=glob-music, 2=atexts, 3=nhs-region)')
    p.add_argument('--is-multi', action='store_true', default=True,
                   help='Use multi-ISI experiment data')
    p.add_argument('--which-isi', type=int, default=None,
                   help='Which ISI (only if not multi)')

    # Encoder
    p.add_argument('--encoder-type', type=str, default='texture_pca',
                   help='Encoder type (texture_pca for 256-PC space)')
    p.add_argument('--pc-dims', type=int, default=256,
                   help='Number of PC dimensions')
    p.add_argument('--device', type=str, default='cuda',
                   help='Device for encoder and score model')

    # Score model
    p.add_argument('--score-config', type=str,
                   default='/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/assets/bryan.yaml',
                   help='Path to score model YAML config')
    p.add_argument('--score-normalize', action='store_true', default=True,
                   help='Normalize score to unit norm')

    # Output
    p.add_argument('--save-dir', type=str,
                   default='reports/figures/prior_guided_grid_search',
                   help='Output directory for results')

    return p.parse_args()


def main():
    print("Starting prior-guided grid search...")
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)

    # ── merge mode ────────────────────────────────────────────────────
    if args.merge:
        print("Merging results...")
        merge_results(args.save_dir)
        return

    print("Setting up grids...")
    # ── grids ─────────────────────────────────────────────────────────
    sigma0_grid = np.array(args.sigma0_grid) if args.sigma0_grid is not None else np.array(DEFAULT_SIGMA0)
    sigma_grid  = np.array(args.sigma_grid)  if args.sigma_grid is not None  else np.array(DEFAULT_SIGMA)
    eta_grid    = np.array(args.eta_grid)    if args.eta_grid is not None    else np.array(DEFAULT_ETA)
    isi_values = tuple(args.isis)

    print("Validating job index...")
    # ── validate job index ────────────────────────────────────────────
    if args.parallel_mode == 'sigma0':
        total_jobs = len(sigma0_grid)
    else:
        total_jobs = len(sigma0_grid) * len(sigma_grid) * len(eta_grid)

    if args.job_index >= total_jobs:
        print(f'ERROR: job-index {args.job_index} >= total jobs {total_jobs}')
        sys.exit(1)

    # ── setup: load real stimuli + encoder ──────────────────────────────
    print(f'Loading experiment data (task={args.which_task}, multi={args.is_multi}) ...')
    exp_list, all_files, name_to_idx, human_runs, task_name, hr_task_name = \
        load_experiment_data(2, args.which_isi, args.is_multi)

    encoder_cfg = dict(
        encoder_type='texture_pca',
        model_name='texture_pca',
        statistics_dict=statistics_set.statistics,
        model_params=model_params,
        pc_dims=args.pc_dims,
        sr=20000,
        duration=2.0,
        rms_level=0.05,
        device=args.device,
    )

    pc_dims = 256

    device = "cuda" if torch.cuda.is_available() else "cpu"

    pc_texture_model = AudioTextureEncoderPCA(
        statistics_dict=statistics_set.statistics,
        pc_dims=pc_dims,
        model_params=model_params,
        sr=20000,
        rms_level=0.01,
        duration=2.0,
        device=device
    )

    print(f'Building prior encoder...')

    encoder = pc_texture_model


    print(f'Encoding {len(all_files)} stimuli ...')
    X0 = encode_stimuli(encoder, all_files)
    print(f'X0 shape: {X0.shape}  (D={X0.shape[1]})')

    # ── setup: load learned score function ────────────────────────────
    print(f'Loading score model from {args.score_config} ...')
    score_model = ScoreFunction(
        mode='textures',
        restart=True,
        config=args.score_config,
        device=args.device,
        normalize=args.score_normalize,
    )
    print('Score model loaded.')

    # ── generate experiment sequences ─────────────────────────────────
    stimulus_pool = sorted({s for seq in exp_list for s in seq})

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
        experiment_list=experiment_list, score_model=score_model,
        isi_values=isi_values, n_mc=args.n_mc,
        seed=args.seed, metric=args.metric,
    )

    print("Dispatching...")
    # ── dispatch ──────────────────────────────────────────────────────
    if args.parallel_mode == 'sigma0':
        _run_sigma0_slice(args, sigma0_grid, sigma_grid, eta_grid,
                          isi_values, common_kwargs)
    else:
        _run_flat_point(args, sigma0_grid, sigma_grid, eta_grid,
                        isi_values, common_kwargs)


def _triple_filename(s0, sig, eta):
    return f's0={s0:.4f}_sig={sig:.4f}_eta={eta:.4f}.npz'


def _run_sigma0_slice(args, sigma0_grid, sigma_grid, eta_grid,
                      isi_values, common_kwargs):
    """Process all (sigma, eta) combos for one sigma0 index."""
    i_s0 = args.job_index
    s0 = sigma0_grid[i_s0]

    n_configs = len(sigma_grid) * len(eta_grid)
    print(f'=== sigma0 slice: idx={i_s0}, sigma0={s0:.4f}, '
          f'{n_configs} configs ===')
    if args.resume:
        print('  --resume: will skip triples with existing per-triple files')

    per_triple_dir = os.path.join(args.save_dir, 'per_triple')
    os.makedirs(per_triple_dir, exist_ok=True)

    results = {isi: np.full((len(sigma_grid), len(eta_grid)), np.nan)
               for isi in isi_values}

    count = 0
    skipped = 0
    t_start = time.perf_counter()

    for i_sig, sig in enumerate(sigma_grid):
        for i_eta, eta in enumerate(eta_grid):
            triple_path = os.path.join(per_triple_dir,
                                       _triple_filename(s0, sig, eta))

            if args.resume and os.path.exists(triple_path):
                existing = np.load(triple_path, allow_pickle=True)
                for isi in isi_values:
                    k = f'dprime_isi{isi}'
                    if k in existing:
                        results[isi][i_sig, i_eta] = float(existing[k])
                skipped += 1
                count += 1
                continue

            dp, triple_data = run_mc_dprime(s0, sig, eta, **common_kwargs)

            for isi in isi_values:
                results[isi][i_sig, i_eta] = dp.get(isi, np.nan)

            np.savez(triple_path, **triple_data)

            count += 1
            if count % 10 == 0 or count == n_configs:
                elapsed = time.perf_counter() - t_start
                rate = count / elapsed if elapsed > 0 else 1
                remaining = (n_configs - count) / rate if rate > 0 else 0
                print(f'  {count}/{n_configs} done  '
                      f'({elapsed:.1f}s elapsed, ~{remaining:.0f}s remaining)')

    t_total = time.perf_counter() - t_start
    print(f'Slice complete: {count} configs in {t_total:.1f}s')
    if skipped:
        print(f'  Skipped (resume): {skipped}')

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
    print(f'Per-triple data saved to {per_triple_dir}/ ({count - skipped} new files)')


def _run_flat_point(args, sigma0_grid, sigma_grid, eta_grid,
                    isi_values, common_kwargs):
    """Process a single (sigma0, sigma, eta) config."""
    shape = (len(sigma0_grid), len(sigma_grid), len(eta_grid))
    total = shape[0] * shape[1] * shape[2]
    if args.job_index >= total:
        print(f'job_index={args.job_index} >= total={total}, nothing to do.')
        return
    i_s0, i_sig, i_eta = np.unravel_index(args.job_index, shape)

    s0 = sigma0_grid[i_s0]
    sig = sigma_grid[i_sig]
    eta = eta_grid[i_eta]

    print(f'=== flat point: idx={args.job_index}, '
          f'sigma0={s0:.4f}, sigma={sig:.4f}, eta={eta:.4f} ===')

    per_triple_dir = os.path.join(args.save_dir, 'per_triple')
    os.makedirs(per_triple_dir, exist_ok=True)

    triple_path = os.path.join(per_triple_dir,
                               _triple_filename(s0, sig, eta))

    if args.resume and os.path.exists(triple_path):
        print(f'Skipping (--resume): {triple_path} already exists')
        existing = np.load(triple_path, allow_pickle=True)
        for isi in isi_values:
            k = f'dprime_isi{isi}'
            val = float(existing[k]) if k in existing else np.nan
            print(f"  ISI={isi}: d'={val:.4f}")
        return

    t_start = time.perf_counter()
    dp, triple_data = run_mc_dprime(s0, sig, eta, **common_kwargs)
    t_total = time.perf_counter() - t_start

    print(f'Done in {t_total:.2f}s')
    for isi in isi_values:
        print(f"  ISI={isi}: d'={dp.get(isi, np.nan):.4f}")

    np.savez(triple_path, **triple_data)
    print(f'Per-triple data saved to {triple_path}')

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
    print("Starting prior-guided grid search...")
    main()
