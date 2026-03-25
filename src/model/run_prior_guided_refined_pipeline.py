#!/usr/bin/env python3
"""
Refinement + single-ISI evaluation pipeline for prior-guided model.

Stage A (multi-ISI refinement)
------------------------------
1) Load coarse/large-search results (merged .npz from run_prior_guided_grid_search.py)
2) Pick top-K configs against full multi-ISI human curve
3) Build dense grids spanning the top-K hyperparameter ranges
4) Run dense search with run_model_core_prior
5) Cross-validate by subsampling human participants/runs
6) Save best models for multi-ISI condition

Stage B (single-ISI transfer)
-----------------------------
Run the best multi-ISI models on single-ISI experiments (default ISI=16) and
save:
- d' at target ISI
- raw trial scores
- item-level hit/FA score dictionaries
for item-wise human/model comparisons.
"""

import argparse
import json
import os
import sys
import time
import types
from collections import defaultdict

import numpy as np
import pandas as pd
import torch

# Mock 'cox' and 'cox.store' so constants.py doesn't crash
cox_mock = types.ModuleType('cox')
store_mock = types.ModuleType('cox.store')
store_mock.PYTORCH_STATE = 'pytorch_state'
cox_mock.store = store_mock
sys.modules['cox'] = cox_mock
sys.modules['cox.store'] = store_mock

# Path setup
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
sys.path.insert(0, _REPO_ROOT)

sys.path.append('/orcd/data/jhm/001/om2/jmhicks/projects/TextureStreaming/code/')
sys.path.append('/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/utls/')
sys.path.append('/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/src/model/')
sys.path.append('/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/')

from texture_prior.params import model_params, statistics_set

from utls.runners_prior import run_model_core_prior
from utls.runners_utils import (
    load_experiment_data,
    build_encoder,
    encode_stimuli,
    compute_human_curve,
)
from utls.toy_experiments import make_high_diversity_sequences
from utls.roc_utils import roc_from_arrays
from utls.analysis_helpers import auroc_to_dprime
from src.model.ScoreFunction import ScoreFunction


DEFAULT_ISIS = [0, 1, 2, 4, 8, 16, 32, 64]


def parse_args():
    p = argparse.ArgumentParser(
        description='Refine prior-guided grid + transfer best models to single-ISI with item scores',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core IO
    p.add_argument('--coarse-results', type=str, required=True,
                   help='Merged coarse grid .npz (grid_search_results_prior_guided.npz)')
    p.add_argument('--save-dir', type=str,
                   default='reports/figures/prior_guided_refined_pipeline',
                   help='Output directory')

    # Multi-ISI setup
    p.add_argument('--which-task', type=int, default=2,
                   help='Task index for multi-ISI fitting (default auditory textures=2)')
    p.add_argument('--isis', type=int, nargs='+', default=DEFAULT_ISIS)
    p.add_argument('--n-sequences', type=int, default=30)
    p.add_argument('--seq-length', type=int, default=120)
    p.add_argument('--min-pairs-per-isi', type=int, default=4)

    # Refinement controls
    p.add_argument('--top-k', type=int, default=15,
                   help='Top coarse models used to define dense ranges')
    p.add_argument('--dense-points', type=int, default=7,
                   help='Points per dimension for dense refinement grid')
    p.add_argument('--n-mc', type=int, default=5,
                   help='MC reps per dense config')

    # CV controls
    p.add_argument('--cv-splits', type=int, default=12,
                   help='Number of random subsample splits for CV scoring')
    p.add_argument('--cv-frac', type=float, default=0.7,
                   help='Fraction of human runs used per split')

    # Single-ISI transfer
    p.add_argument('--single-isi', type=int, default=16,
                   help='Single-ISI condition used for transfer checks')
    p.add_argument('--single-isi-tasks', type=int, nargs='+', default=[0, 1, 2, 3],
                   help='Task indices to evaluate in single-ISI mode')
    p.add_argument('--n-best-transfer', type=int, default=10,
                   help='How many refined best models to transfer to single-ISI')

    # Model stack
    p.add_argument('--pc-dims', type=int, default=256)
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--metric', type=str, default='cosine')
    p.add_argument('--score-config', type=str,
                   default='/om2/user/bjmedina/auditory-memory/memory/assets/bryan.yaml')
    p.add_argument('--score-normalize', action='store_true', default=True)

    p.add_argument('--seed', type=int, default=44)
    return p.parse_args()


def _safe_geomspace(lo, hi, n):
    """Log-spaced vector robust to zeros.

    If lo==hi returns constant array. If any bound <=0, falls back to linear.
    """
    if np.isclose(lo, hi):
        return np.full(n, float(lo))
    if lo > 0 and hi > 0:
        return np.geomspace(lo, hi, n)
    return np.linspace(lo, hi, n)


def load_encoder_and_score(args):
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

    print('Building texture PCA encoder...')
    encoder = build_encoder(encoder_cfg)

    print('Loading score model...')
    score_model = ScoreFunction(
        mode='textures',
        restart=True,
        config=args.score_config,
        device=args.device,
        normalize=args.score_normalize,
    )
    return encoder, score_model


def compute_model_dprime_curve(run_out, isis):
    """Compute d' per ISI from a run_model_core_prior output."""
    dprimes = []
    fas = np.asarray(run_out['fas'], float)
    score_type = run_out.get('score_type', 'distance')

    for isi in isis:
        hits_raw = run_out['isi_hit_dists'].get(isi + 1, [])
        hits = np.asarray([s for s, _ in hits_raw], float)
        if len(hits) < 3 or len(fas) < 3:
            dprimes.append(np.nan)
            continue
        roc = roc_from_arrays(hits, fas, score_type=score_type)
        if roc is None:
            dprimes.append(np.nan)
            continue
        dprimes.append(float(auroc_to_dprime(roc[2])))
    return np.asarray(dprimes, float)


def run_mc_curve(sigma0, sigma, eta, *, X0, name_to_idx, experiment_list,
                 score_model, isis, n_mc, metric, seed):
    all_hits = defaultdict(list)
    all_fas = []

    for rep in range(n_mc):
        run = run_model_core_prior(
            sigma0=sigma0,
            sigma=sigma,
            X0=X0,
            name_to_idx=name_to_idx,
            experiment_list=experiment_list,
            score_model=score_model,
            drift_step_size=eta,
            metric=metric,
            seed=seed * 10_000 + rep,
        )
        for isi in isis:
            all_hits[isi].extend(run['isi_hit_dists'].get(isi + 1, []))
        all_fas.extend(run['fas'])

    fas = np.asarray(all_fas, float)
    dprimes = []
    for isi in isis:
        hits = np.asarray([s for s, _ in all_hits[isi]], float)
        if len(hits) < 3 or len(fas) < 3:
            dprimes.append(np.nan)
            continue
        roc = roc_from_arrays(hits, fas, score_type='distance' if metric != 'loglikelihood' else 'likelihood')
        dprimes.append(float(auroc_to_dprime(roc[2])) if roc is not None else np.nan)
    return np.asarray(dprimes, float)


def mse(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if not np.any(m):
        return np.nan
    return float(np.mean((a[m] - b[m]) ** 2))


def score_against_subsampled_humans(model_curve, human_runs, *, is_multi,
                                    which_isi, cv_splits, cv_frac, seed):
    """Cross-validated fit score via repeated subsamples of participant runs."""
    rng = np.random.default_rng(seed)
    n = len(human_runs)
    k = max(2, int(np.ceil(n * cv_frac)))

    vals = []
    for _ in range(cv_splits):
        idx = rng.choice(n, size=min(k, n), replace=False)
        subset = [human_runs[i] for i in idx]
        h_curve = compute_human_curve(subset, is_multi=is_multi, which_isi=which_isi)
        vals.append(mse(model_curve, h_curve))

    vals = np.asarray(vals, float)
    return {
        'cv_mse_mean': float(np.nanmean(vals)),
        'cv_mse_std': float(np.nanstd(vals)),
        'cv_mse_all': vals,
    }


def dense_grid_from_top_k(coarse_npz, human_curve, isis, top_k, dense_points):
    """Pick top-K coarse configs then build dense refinement ranges."""
    sigma0_grid = coarse_npz['sigma0_grid']
    sigma_grid = coarse_npz['sigma_grid']
    eta_grid = coarse_npz['eta_grid']

    dprime_stack = np.stack([coarse_npz[f'dprime_isi{isi}'] for isi in isis], axis=-1)
    flat = dprime_stack.reshape(-1, len(isis))

    scores = np.array([mse(row, human_curve) for row in flat], float)
    order = np.argsort(scores)
    top_idx = order[:min(top_k, len(order))]

    i_s0, i_sig, i_eta = np.unravel_index(top_idx, dprime_stack.shape[:3])
    top_sigma0 = sigma0_grid[i_s0]
    top_sigma = sigma_grid[i_sig]
    top_eta = eta_grid[i_eta]

    dense_sigma0 = _safe_geomspace(np.min(top_sigma0), np.max(top_sigma0), dense_points)
    dense_sigma = _safe_geomspace(np.min(top_sigma), np.max(top_sigma), dense_points)
    dense_eta = _safe_geomspace(np.min(top_eta), np.max(top_eta), dense_points)

    return {
        'top_indices_flat': top_idx,
        'top_scores_mse': scores[top_idx],
        'top_sigma0': top_sigma0,
        'top_sigma': top_sigma,
        'top_eta': top_eta,
        'dense_sigma0': dense_sigma0,
        'dense_sigma': dense_sigma,
        'dense_eta': dense_eta,
    }


def run_dense_refinement(args, *, X0, name_to_idx, experiment_list,
                         score_model, human_runs):
    print(f'Loading coarse results: {args.coarse_results}')
    coarse = np.load(args.coarse_results)

    isis = np.array(args.isis, dtype=int)
    human_curve = compute_human_curve(human_runs, is_multi=True, which_isi=None)

    dense_info = dense_grid_from_top_k(
        coarse_npz=coarse,
        human_curve=human_curve,
        isis=isis,
        top_k=args.top_k,
        dense_points=args.dense_points,
    )

    dense_sigma0 = dense_info['dense_sigma0']
    dense_sigma = dense_info['dense_sigma']
    dense_eta = dense_info['dense_eta']

    print('Dense ranges:')
    print(f'  sigma0: [{dense_sigma0[0]:.4g}, {dense_sigma0[-1]:.4g}] ({len(dense_sigma0)} pts)')
    print(f'  sigma : [{dense_sigma[0]:.4g}, {dense_sigma[-1]:.4g}] ({len(dense_sigma)} pts)')
    print(f'  eta   : [{dense_eta[0]:.4g}, {dense_eta[-1]:.4g}] ({len(dense_eta)} pts)')

    rows = []
    total = len(dense_sigma0) * len(dense_sigma) * len(dense_eta)
    done = 0
    t0 = time.perf_counter()

    for s0 in dense_sigma0:
        for sig in dense_sigma:
            for eta in dense_eta:
                curve = run_mc_curve(
                    sigma0=float(s0),
                    sigma=float(sig),
                    eta=float(eta),
                    X0=X0,
                    name_to_idx=name_to_idx,
                    experiment_list=experiment_list,
                    score_model=score_model,
                    isis=isis,
                    n_mc=args.n_mc,
                    metric=args.metric,
                    seed=args.seed,
                )

                full_mse = mse(curve, human_curve)
                cv = score_against_subsampled_humans(
                    curve,
                    human_runs,
                    is_multi=True,
                    which_isi=None,
                    cv_splits=args.cv_splits,
                    cv_frac=args.cv_frac,
                    seed=args.seed + done,
                )

                row = {
                    'sigma0': float(s0),
                    'sigma': float(sig),
                    'eta': float(eta),
                    'mse_full': full_mse,
                    'cv_mse_mean': cv['cv_mse_mean'],
                    'cv_mse_std': cv['cv_mse_std'],
                }
                for isi, dp in zip(isis, curve):
                    row[f'dprime_isi{int(isi)}'] = float(dp)
                rows.append(row)

                done += 1
                if done % 25 == 0 or done == total:
                    elapsed = time.perf_counter() - t0
                    print(f'  dense {done}/{total} ({elapsed:.1f}s)')

    df = pd.DataFrame(rows).sort_values(['cv_mse_mean', 'mse_full'], ascending=True)

    os.makedirs(args.save_dir, exist_ok=True)
    dense_csv = os.path.join(args.save_dir, 'dense_refined_results.csv')
    df.to_csv(dense_csv, index=False)

    top_df = df.head(args.n_best_transfer).copy()
    top_json = os.path.join(args.save_dir, 'best_models_multi_isi.json')
    with open(top_json, 'w') as f:
        json.dump(top_df.to_dict(orient='records'), f, indent=2)

    meta = {
        'coarse_results': args.coarse_results,
        'top_k': args.top_k,
        'dense_points': args.dense_points,
        'n_mc': args.n_mc,
        'cv_splits': args.cv_splits,
        'cv_frac': args.cv_frac,
        'isis': [int(v) for v in isis],
        'dense_range_sigma0': [float(dense_sigma0[0]), float(dense_sigma0[-1])],
        'dense_range_sigma': [float(dense_sigma[0]), float(dense_sigma[-1])],
        'dense_range_eta': [float(dense_eta[0]), float(dense_eta[-1])],
    }
    with open(os.path.join(args.save_dir, 'dense_refined_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    print(f'Saved dense refinement CSV: {dense_csv}')
    print(f'Saved best-model list: {top_json}')
    return df


def run_single_isi_transfer(args, *, encoder, score_model, best_df):
    out_dir = os.path.join(args.save_dir, f'single_isi_{args.single_isi}')
    os.makedirs(out_dir, exist_ok=True)

    task_rows = []
    for task_idx in args.single_isi_tasks:
        print(f'Loading single-ISI data: task={task_idx}, isi={args.single_isi}')
        exp_list, all_files, name_to_idx, human_runs, task_name, hr_task_name = load_experiment_data(
            which_task=task_idx,
            which_isi=args.single_isi,
            is_multi=False,
        )
        X0 = encode_stimuli(encoder, all_files).float().to(args.device)

        for model_rank, (_, rec) in enumerate(best_df.iterrows(), start=1):
            run = run_model_core_prior(
                sigma0=float(rec['sigma0']),
                sigma=float(rec['sigma']),
                X0=X0,
                name_to_idx=name_to_idx,
                experiment_list=exp_list,
                score_model=score_model,
                drift_step_size=float(rec['eta']),
                metric=args.metric,
                seed=args.seed + model_rank + task_idx * 100,
                return_item_scores=True,
                return_trial_log=True,
            )

            target_key = args.single_isi + 1
            hits = np.asarray([s for s, _ in run['isi_hit_dists'].get(target_key, [])], float)
            fas = np.asarray(run['fas'], float)
            if len(hits) >= 3 and len(fas) >= 3:
                roc = roc_from_arrays(hits, fas, score_type=run.get('score_type', 'distance'))
                dprime_target = float(auroc_to_dprime(roc[2])) if roc is not None else np.nan
            else:
                dprime_target = np.nan

            model_id = f'rank{model_rank:02d}_task{task_idx}'
            trial_df = pd.DataFrame(run.get('trial_log', []))
            trial_path = os.path.join(out_dir, f'{model_id}_trial_scores.csv')
            trial_df.to_csv(trial_path, index=False)

            item_payload = {
                'item_hits': run.get('item_hits', {}),
                'item_fas': run.get('item_fas', {}),
            }
            item_path = os.path.join(out_dir, f'{model_id}_item_scores.json')
            with open(item_path, 'w') as f:
                json.dump(item_payload, f)

            npz_path = os.path.join(out_dir, f'{model_id}_run_summary.npz')
            np.savez(
                npz_path,
                sigma0=float(rec['sigma0']),
                sigma=float(rec['sigma']),
                eta=float(rec['eta']),
                task_idx=int(task_idx),
                task_name=task_name,
                hr_task_name=hr_task_name,
                target_isi=int(args.single_isi),
                dprime_target=float(dprime_target),
                hit_scores_target=hits,
                fa_scores=fas,
            )

            task_rows.append({
                'task_idx': int(task_idx),
                'task_name': task_name,
                'hr_task_name': hr_task_name,
                'model_rank': model_rank,
                'sigma0': float(rec['sigma0']),
                'sigma': float(rec['sigma']),
                'eta': float(rec['eta']),
                'single_isi': int(args.single_isi),
                'dprime_isi_target': dprime_target,
                'trial_scores_csv': trial_path,
                'item_scores_json': item_path,
                'run_summary_npz': npz_path,
                'n_human_runs_single_isi': len(human_runs),
            })

    summary = pd.DataFrame(task_rows)
    summary_csv = os.path.join(out_dir, 'single_isi_transfer_summary.csv')
    summary.to_csv(summary_csv, index=False)
    print(f'Saved single-ISI transfer summary: {summary_csv}')


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    print(args)

    # Shared representations/score model
    encoder, score_model = load_encoder_and_score(args)

    # Multi-ISI task data for fitting
    exp_list, all_files, name_to_idx, human_runs, task_name, hr_task_name = load_experiment_data(
        which_task=args.which_task,
        which_isi=None,
        is_multi=True,
    )
    print(f'Multi-ISI fitting task: {task_name} ({hr_task_name})')

    print(f'Encoding stimuli once for all stages (N={len(all_files)})...')
    X0 = encode_stimuli(encoder, all_files).float().to(args.device)
    print(f'X0 shape: {tuple(X0.shape)}')

    stimulus_pool = sorted({s for seq in exp_list for s in seq})
    experiment_list, _ = make_high_diversity_sequences(
        stimulus_pool=stimulus_pool,
        isi_values=list(args.isis),
        n_sequences=args.n_sequences,
        length=args.seq_length,
        min_pairs_per_isi=args.min_pairs_per_isi,
        seed=args.seed,
    )

    # Stage A: dense refinement + CV selection
    dense_df = run_dense_refinement(
        args,
        X0=X0,
        name_to_idx=name_to_idx,
        experiment_list=experiment_list,
        score_model=score_model,
        human_runs=human_runs,
    )

    # Stage B: run best models on single-ISI experiments + save item-level scores
    best_df = dense_df.head(args.n_best_transfer).copy()
    run_single_isi_transfer(args, encoder=encoder, score_model=score_model, best_df=best_df)

    print('Done.')


if __name__ == '__main__':
    main()
