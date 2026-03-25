#!/usr/bin/env python3
"""Unified fine-grid refinement pipeline for prior-guided and 3-step models."""

import argparse
import os
import sys
import types
from collections import Counter, defaultdict

import numpy as np

# Mock 'cox' and 'cox.store' so constants.py doesn't crash
cox_mock = types.ModuleType('cox')
store_mock = types.ModuleType('cox.store')
store_mock.PYTORCH_STATE = 'pytorch_state'
cox_mock.store = store_mock
sys.modules['cox'] = cox_mock
sys.modules['cox.store'] = store_mock

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..'))
sys.path.insert(0, _REPO_ROOT)

sys.path.append('/orcd/data/jhm/001/om2/jmhicks/projects/TextureStreaming/code/')
sys.path.append('/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/utls/')
sys.path.append('/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/src/model/')
sys.path.append('/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/')

from texture_prior.params import model_params, statistics_set

from src.model.ScoreFunction import ScoreFunction
from src.model.run_3step_grid_search import run_mc_dprime as run_mc_dprime_3step
from src.model.run_3step_grid_search import merge_results as merge_results_3step
from src.model.run_prior_guided_grid_search import run_mc_dprime as run_mc_dprime_prior
from src.model.run_prior_guided_grid_search import merge_results as merge_results_prior
from utls.analysis_helpers import auroc_to_dprime, roc_for_isi
from utls.roc_utils import roc_from_arrays
from utls.runners_prior import run_model_core_prior
from utls.runners_utils import (
    build_encoder,
    compute_mse,
    cosine_similarity_dp,
    encode_stimuli,
    load_experiment_data,
)
from utls.runners_v2 import make_noise_schedule, run_model_core
from utls.toy_experiments import make_high_diversity_sequences

DEFAULT_ISIS = [0, 1, 2, 4, 8, 16, 32, 64]

MODEL_CONFIGS = {
    'prior': {
        'param_names': ('sigma0', 'sigma', 'eta'),
        'grid_keys': ('sigma0_grid', 'sigma_grid', 'eta_grid'),
        'merged_file': 'grid_search_results_prior_guided.npz',
        'fine_merged_file': 'grid_search_results_prior_guided_fine.npz',
        'default_which_task': 2,
        'encoder_type': 'texture_pca',
        'needs_score_model': True,
    },
    '3step': {
        'param_names': ('sigma0', 'sigma1', 'sigma2'),
        'grid_keys': ('sigma0_grid', 'sigma1_grid', 'sigma2_grid'),
        'merged_file': 'grid_search_results_3step_t5.npz',
        'fine_merged_file': 'grid_search_results_3step_fine.npz',
        'default_which_task': 0,
        'encoder_type': 'resnet50',
        'needs_score_model': False,
    },
}


def _human_curve_from_runs(human_runs, isi_values):
    vals = []
    for isi in isi_values:
        aucs = []
        for run in human_runs:
            res = roc_for_isi(run, isi)
            if res is not None:
                _, _, auc = res
                aucs.append(auroc_to_dprime(auc))
        vals.append(np.nanmean(aucs) if aucs else np.nan)
    return np.asarray(vals, dtype=float)


def _flatten_model_dprimes(data, grid_keys, isi_values):
    shape = tuple(len(data[k]) for k in grid_keys)
    stacked = np.stack([data[f'dprime_isi{isi}'] for isi in isi_values], axis=-1)
    flat = stacked.reshape(-1, len(isi_values))
    idx_tuples = np.array(np.unravel_index(np.arange(flat.shape[0]), shape)).T
    return flat, idx_tuples


def _row_to_param_dict(irow, data, model_type):
    cfg = MODEL_CONFIGS[model_type]
    out = {}
    for pname, gk, gi in zip(cfg['param_names'], cfg['grid_keys'], irow):
        out[pname] = float(data[gk][int(gi)])
    return out


def load_encoder_and_score(model_type, device='cuda', pc_dims=256, which_task=None):
    cfg = MODEL_CONFIGS[model_type]
    if model_type == 'prior':
        encoder_cfg = dict(
            encoder_type='texture_pca',
            model_name='texture_pca',
            statistics_dict=statistics_set.statistics,
            model_params=model_params,
            pc_dims=pc_dims,
            sr=20000,
            duration=2.0,
            rms_level=0.05,
            device=device,
        )
        encoder = build_encoder(encoder_cfg)
        score_model = ScoreFunction(mode='textures', restart=True, device=device)
        return encoder, score_model

    encoder_cfg = dict(
        encoder_type='resnet50',
        model_name='resnet50',
        layer='layer4',
        pc_dims=None,
        device=device,
    )
    encoder = build_encoder(encoder_cfg)
    return encoder, None


def dense_grid_from_top_k(
    coarse_results_path,
    human_runs,
    *,
    model_type,
    top_k=15,
    n_points=20,
    n_folds=10,
    seed=42,
    is_multi=True,
    which_isi=None,
):
    if not is_multi:
        raise ValueError('dense_grid_from_top_k expects multi-ISI runs (is_multi=True).')
    cfg = MODEL_CONFIGS[model_type]

    data = np.load(coarse_results_path)
    isi_values = [int(x) for x in data['isi_values']]
    all_model_dprimes, idx_tuples = _flatten_model_dprimes(data, cfg['grid_keys'], isi_values)

    n_models = all_model_dprimes.shape[0]
    fold_mse = [[] for _ in range(n_models)]
    fold_cos = [[] for _ in range(n_models)]
    topk_counter = Counter()

    rng = np.random.default_rng(seed)
    subjects = np.arange(len(human_runs))

    for _ in range(n_folds):
        perm = rng.permutation(subjects)
        half = perm[: max(1, len(perm) // 2)]
        fold_runs = [human_runs[int(i)] for i in half]
        fold_human_dp = _human_curve_from_runs(fold_runs, isi_values)

        mses = np.array([compute_mse(mdp, fold_human_dp) for mdp in all_model_dprimes], dtype=float)
        coss = np.array([cosine_similarity_dp(mdp, fold_human_dp, start_idx=0) for mdp in all_model_dprimes], dtype=float)

        for i in range(n_models):
            fold_mse[i].append(mses[i])
            fold_cos[i].append(coss[i])

        valid_idx = np.where(np.isfinite(mses))[0]
        order = valid_idx[np.argsort(mses[valid_idx])]
        for i in order[: min(top_k, len(order))]:
            topk_counter[int(i)] += 1

    min_count = max(1, int(np.ceil(n_folds * 0.5)))
    robust = sorted([i for i, c in topk_counter.items() if c >= min_count])

    if not robust:
        mean_mse = np.array([np.nanmean(v) for v in fold_mse])
        valid_idx = np.where(np.isfinite(mean_mse))[0]
        robust = [int(i) for i in valid_idx[np.argsort(mean_mse[valid_idx])[: max(1, top_k)]]]

    robust_params = [_row_to_param_dict(idx_tuples[i], data, model_type) for i in robust]

    param_ranges = {}
    out = {}
    for pname, gkey in zip(cfg['param_names'], cfg['grid_keys']):
        vals = np.array([p[pname] for p in robust_params], dtype=float)
        lo, hi = float(np.min(vals)), float(np.max(vals))
        param_ranges[pname] = (lo, hi)
        out[gkey] = np.linspace(lo, hi, n_points, dtype=float)

    out.update(
        {
            'top_k_params': robust_params,
            'cv_mse_scores': np.array([np.nanmean(v) for v in fold_mse], dtype=float),
            'cv_cosine_scores': np.array([np.nanmean(v) for v in fold_cos], dtype=float),
            'param_ranges': param_ranges,
            'model_type': model_type,
            'isi_values': np.array(isi_values, dtype=int),
        }
    )
    return out


def run_dense_refinement(
    dense_grid_path,
    *,
    model_type,
    X0,
    name_to_idx,
    experiment_list,
    score_model,
    isi_values,
    n_mc,
    seed,
    metric,
    job_index,
    parallel_mode,
    save_dir,
    resume=False,
    t_step=5,
):
    os.makedirs(save_dir, exist_ok=True)
    dense = np.load(dense_grid_path, allow_pickle=True)
    cfg = MODEL_CONFIGS[model_type]

    g0, g1, g2 = [dense[k] for k in cfg['grid_keys']]

    if parallel_mode == 'sigma0':
        i0 = int(job_index)
        if i0 < 0 or i0 >= len(g0):
            raise IndexError(f'job-index {job_index} out of range for sigma0 mode')

        out_path = os.path.join(save_dir, f'grid_slice_s0idx{i0:03d}.npz')
        if resume and os.path.exists(out_path):
            return out_path

        vals_by_isi = {int(isi): np.full((len(g1), len(g2)), np.nan, dtype=float) for isi in isi_values}

        for i1, v1 in enumerate(g1):
            for i2, v2 in enumerate(g2):
                if model_type == 'prior':
                    dps, _ = run_mc_dprime_prior(
                        float(g0[i0]), float(v1), float(v2),
                        X0=X0, name_to_idx=name_to_idx, experiment_list=experiment_list,
                        score_model=score_model, isi_values=isi_values,
                        n_mc=n_mc, seed=seed, metric=metric,
                    )
                else:
                    dps, _ = run_mc_dprime_3step(
                        float(g0[i0]), float(v1), float(v2),
                        X0=X0, name_to_idx=name_to_idx, experiment_list=experiment_list,
                        t_step=t_step, isi_values=isi_values,
                        n_mc=n_mc, seed=seed, metric=metric,
                    )
                for isi in isi_values:
                    vals_by_isi[int(isi)][i1, i2] = float(dps.get(int(isi), np.nan))

        np.savez(
            out_path,
            **{cfg['grid_keys'][0]: g0, cfg['grid_keys'][1]: g1, cfg['grid_keys'][2]: g2},
            isi_values=np.array(isi_values, dtype=int),
            sigma0_idx=i0,
            **{f'dprime_isi{isi}': vals_by_isi[int(isi)] for isi in isi_values},
        )
        return out_path

    total = len(g0) * len(g1) * len(g2)
    flat_idx = int(job_index)
    if flat_idx < 0 or flat_idx >= total:
        raise IndexError(f'job-index {job_index} out of range for flat mode total={total}')

    i0 = flat_idx // (len(g1) * len(g2))
    rem = flat_idx % (len(g1) * len(g2))
    i1 = rem // len(g2)
    i2 = rem % len(g2)

    out_path = os.path.join(save_dir, f'grid_point_{i0:03d}_{i1:03d}_{i2:03d}.npz')
    if resume and os.path.exists(out_path):
        return out_path

    p0, p1, p2 = float(g0[i0]), float(g1[i1]), float(g2[i2])
    if model_type == 'prior':
        dps, triple_data = run_mc_dprime_prior(
            p0, p1, p2,
            X0=X0, name_to_idx=name_to_idx, experiment_list=experiment_list,
            score_model=score_model, isi_values=isi_values,
            n_mc=n_mc, seed=seed, metric=metric,
        )
    else:
        dps, triple_data = run_mc_dprime_3step(
            p0, p1, p2,
            X0=X0, name_to_idx=name_to_idx, experiment_list=experiment_list,
            t_step=t_step, isi_values=isi_values,
            n_mc=n_mc, seed=seed, metric=metric,
        )

    np.savez(
        out_path,
        **{cfg['grid_keys'][0]: g0, cfg['grid_keys'][1]: g1, cfg['grid_keys'][2]: g2},
        **{f'{cfg["param_names"][0]}_idx': i0, f'{cfg["param_names"][1]}_idx': i1, f'{cfg["param_names"][2]}_idx': i2},
        **{f'dprime_isi{isi}': float(dps.get(int(isi), np.nan)) for isi in isi_values},
        **triple_data,
    )
    return out_path


def run_single_isi_transfer(
    best_params_list,
    *,
    model_type,
    which_task,
    which_isi=16,
    n_mc=10,
    seed=44,
    metric='cosine',
    save_dir,
    device='cuda',
    t_step=5,
):
    encoder, score_model = load_encoder_and_score(model_type=model_type, device=device, which_task=which_task)
    experiment_list, all_files, name_to_idx, _, _, _ = load_experiment_data(
        which_task=which_task,
        which_isi=which_isi,
        is_multi=False,
    )

    X0 = encode_stimuli(encoder, all_files).float().to(device)

    stimulus_pool = sorted({s for seq in experiment_list for s in seq})
    seqs, _ = make_high_diversity_sequences(
        stimulus_pool=stimulus_pool,
        isi_values=[0, which_isi],
        n_sequences=30,
        length=120,
        min_pairs_per_isi=4,
        seed=seed,
    )

    os.makedirs(save_dir, exist_ok=True)
    score_type = 'distance' if metric != 'loglikelihood' else 'likelihood'

    for rank, params in enumerate(best_params_list, start=1):
        item_hits = defaultdict(list)
        item_fas = defaultdict(list)
        all_hits16 = []
        all_fas = []

        for rep in range(n_mc):
            rep_seed = seed * 10_000 + rep + rank * 100

            if model_type == 'prior':
                run = run_model_core_prior(
                    sigma0=float(params['sigma0']),
                    sigma=float(params['sigma']),
                    X0=X0,
                    name_to_idx=name_to_idx,
                    experiment_list=seqs,
                    score_model=score_model,
                    drift_step_size=float(params['eta']),
                    metric=metric,
                    seed=rep_seed,
                    return_item_scores=True,
                )
                hits16 = [s for s, t in run['isi_hit_dists'].get(which_isi + 1, []) if (t > 0)]
                all_hits16.extend(hits16)
                all_fas.extend(run['fas'])
                for k, v in run.get('item_hits', {}).items():
                    item_hits[k].extend(v)
                for k, v in run.get('item_fas', {}).items():
                    item_fas[k].extend(v)
            else:
                schedule = make_noise_schedule('three-regime', {
                    'sigma0': float(params['sigma0']),
                    'sigma1': float(params['sigma1']),
                    'sigma2': float(params['sigma2']),
                    't_step': t_step,
                })
                run_trial = run_model_core(
                    sigma0=float(params['sigma0']),
                    X0=X0,
                    name_to_idx=name_to_idx,
                    experiment_list=seqs,
                    metric=metric,
                    noise_schedule=schedule,
                    seed=rep_seed,
                    return_item_scores=False,
                )
                run_item = run_model_core(
                    sigma0=float(params['sigma0']),
                    X0=X0,
                    name_to_idx=name_to_idx,
                    experiment_list=seqs,
                    metric=metric,
                    noise_schedule=schedule,
                    seed=rep_seed,
                    return_item_scores=True,
                )
                hits16 = [s for s, t in run_trial['isi_hit_dists'].get(which_isi + 1, []) if (t > 0)]
                all_hits16.extend(hits16)
                all_fas.extend(run_trial['fas'])
                for k, v in run_item.get('item_hits', {}).items():
                    item_hits[k].extend(v)
                for k, v in run_item.get('item_fas', {}).items():
                    item_fas[k].extend(v)

        hits_arr = np.asarray(all_hits16, dtype=float)
        fas_arr = np.asarray(all_fas, dtype=float)
        roc = roc_from_arrays(hits_arr, fas_arr, score_type=score_type) if (len(hits_arr) >= 3 and len(fas_arr) >= 3) else None
        dprime16 = float(auroc_to_dprime(roc[2])) if roc is not None else np.nan

        out_name = f'transfer_isi{which_isi}_{model_type}_rank{rank:02d}.npz'
        np.savez(
            os.path.join(save_dir, out_name),
            model_type=model_type,
            which_task=int(which_task),
            which_isi=int(which_isi),
            dprime_isi=float(dprime16),
            hit_scores_isi=hits_arr,
            fa_scores=fas_arr,
            item_hits=np.array(dict(item_hits), dtype=object),
            item_fas=np.array(dict(item_fas), dtype=object),
            raw_params=np.array(params, dtype=object),
        )


def _save_dense_grid_spec(path, dense):
    payload = {}
    for k, v in dense.items():
        if isinstance(v, dict) or isinstance(v, list):
            payload[k] = np.array(v, dtype=object)
        else:
            payload[k] = v
    np.savez(path, **payload)


def _load_best_params_from_dense_spec(path, model_type, top_n=10):
    dense = np.load(path, allow_pickle=True)
    cfg = MODEL_CONFIGS[model_type]
    g0, g1, g2 = [dense[k] for k in cfg['grid_keys']]
    mses = np.asarray(dense['cv_mse_scores'], dtype=float)
    idx = np.where(np.isfinite(mses))[0]
    order = idx[np.argsort(mses[idx])][: min(top_n, len(idx))]

    shape = (len(g0), len(g1), len(g2))
    unr = np.array(np.unravel_index(order, shape)).T

    params = []
    for i0, i1, i2 in unr:
        params.append(
            {
                cfg['param_names'][0]: float(g0[i0]),
                cfg['param_names'][1]: float(g1[i1]),
                cfg['param_names'][2]: float(g2[i2]),
            }
        )
    return params


def parse_args():
    p = argparse.ArgumentParser(description='Unified refined pipeline (prior or 3step)')
    p.add_argument('--model-type', choices=['prior', '3step'], required=True)
    p.add_argument('--mode', choices=['dense-grid', 'fine-search', 'merge-fine', 'transfer-isi16', 'full'], default='full')

    p.add_argument('--coarse-results-path', type=str, default='')
    p.add_argument('--save-dir', type=str, default='reports/figures/refined_pipeline')
    p.add_argument('--dense-grid-path', type=str, default='')

    p.add_argument('--job-index', type=int, default=0)
    p.add_argument('--parallel-mode', choices=['flat', 'sigma0'], default='flat')
    p.add_argument('--top-k', type=int, default=15)
    p.add_argument('--n-points', type=int, default=20)
    p.add_argument('--n-folds', type=int, default=10)
    p.add_argument('--n-mc', type=int, default=10)
    p.add_argument('--which-task', type=int, default=None)
    p.add_argument('--metric', type=str, default='cosine')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--t-step', type=int, default=5)
    p.add_argument('--resume', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    cfg = MODEL_CONFIGS[args.model_type]
    which_task = cfg['default_which_task'] if args.which_task is None else args.which_task

    os.makedirs(args.save_dir, exist_ok=True)

    coarse_path = args.coarse_results_path or os.path.join(args.save_dir, cfg['merged_file'])
    dense_path = args.dense_grid_path or os.path.join(args.save_dir, f'dense_grid_spec_{args.model_type}.npz')

    if args.mode in ('dense-grid', 'full'):
        _, _, _, human_runs, _, _ = load_experiment_data(which_task=which_task, which_isi=None, is_multi=True)
        dense = dense_grid_from_top_k(
            coarse_path,
            human_runs,
            model_type=args.model_type,
            top_k=args.top_k,
            n_points=args.n_points,
            n_folds=args.n_folds,
            seed=args.seed,
            is_multi=True,
        )
        _save_dense_grid_spec(dense_path, dense)
        print(f'Saved dense grid spec to {dense_path}')

    if args.mode in ('fine-search', 'full'):
        encoder, score_model = load_encoder_and_score(args.model_type, device=args.device, which_task=which_task)
        experiment_list, all_files, name_to_idx, _, _, _ = load_experiment_data(
            which_task=which_task,
            which_isi=None,
            is_multi=True,
        )
        X0 = encode_stimuli(encoder, all_files).float().to(args.device)

        out_path = run_dense_refinement(
            dense_path,
            model_type=args.model_type,
            X0=X0,
            name_to_idx=name_to_idx,
            experiment_list=experiment_list,
            score_model=score_model,
            isi_values=DEFAULT_ISIS,
            n_mc=args.n_mc,
            seed=args.seed,
            metric=args.metric,
            job_index=args.job_index,
            parallel_mode=args.parallel_mode,
            save_dir=args.save_dir,
            resume=args.resume,
            t_step=args.t_step,
        )
        print(f'Saved fine-search output to {out_path}')

    if args.mode in ('merge-fine', 'full'):
        if args.model_type == 'prior':
            merge_results_prior(args.save_dir)
        else:
            merge_results_3step(args.save_dir)

    if args.mode in ('transfer-isi16', 'full'):
        best_params = _load_best_params_from_dense_spec(dense_path, model_type=args.model_type, top_n=10)
        run_single_isi_transfer(
            best_params,
            model_type=args.model_type,
            which_task=which_task,
            which_isi=16,
            n_mc=args.n_mc,
            seed=args.seed,
            metric=args.metric,
            save_dir=args.save_dir,
            device=args.device,
            t_step=args.t_step,
        )
        print('Saved transfer outputs.')


if __name__ == '__main__':
    main()
