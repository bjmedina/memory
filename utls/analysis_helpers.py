#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utls.analysis_helpers — orchestration and cross-noise utilities
"""

import numpy as np
from collections import defaultdict
from sklearn.metrics import roc_curve
from utls.roc_utils import roc_from_arrays

from scipy.stats import norm

from sklearn.utils import resample
from sklearn.linear_model import LinearRegression

def find_optimal_roc_threshold(hits, fas, score_type="distance"):
    """
    Given hit and false-alarm scores, return the decision threshold
    corresponding to the ROC point closest to (0,1).
    """
    y_true = np.concatenate([np.ones(len(hits)), np.zeros(len(fas))])
    y_score = np.concatenate([hits, fas])
    if score_type == "distance":  # smaller distance = more 'yes'
        y_score = -y_score

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    dists = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
    i_best = np.argmin(dists)

    return {
        "threshold": thresholds[i_best],
        "fpr": fpr[i_best],
        "tpr": tpr[i_best],
        "distance": dists[i_best]
    }

def compute_model_dprime_curve(run_data):
    """Compute d′ vs ISI for this model run."""
    isi_vals = sorted(run_data["isi_hit_dists"].keys())
    dprimes = []
    for isi in isi_vals:
        res = roc_for_isi(run_data, isi)
        if res is not None:
            _, _, auc = res
            dprimes.append(auroc_to_dprime(auc))
        else:
            dprimes.append(np.nan)
    return np.array(isi_vals, float), np.array(dprimes, float)
    
def bootstrap_dprime_ci(run_data, isi_value, n_boot=200, ci=68):
    """
    Compute bootstrap mean and SEM for d′ at a given ISI.
    
    Args:
        run_data : dict with "isi_hit_dists" and "fas"
        isi_value : ISI to analyze
        n_boot : number of bootstrap resamples
        ci : confidence interval percentage (default 68 ≈ 1 SEM)
    Returns:
        mean_dprime, sem_dprime
    """
    hits_raw = run_data["isi_hit_dists"].get(isi_value, [])
    if not hits_raw:
        return np.nan, np.nan

    hits = np.array([d for (d, t) in hits_raw], float)
    fas  = np.asarray(run_data["fas"], float)
    if len(hits) < 3 or len(fas) < 3:
        return np.nan, np.nan

    score_type = run_data.get("score_type", "distance")
    dprimes = []

    for _ in range(n_boot):
        h_bs = resample(hits, replace=True)
        f_bs = resample(fas, replace=True)
        res = roc_from_arrays(h_bs, f_bs, score_type=score_type)
        if res is not None:
            _, _, auc_val = res
            dprimes.append(auroc_to_dprime(auc_val))

    dprimes = np.array(dprimes)
    mean_d = np.nanmean(dprimes)
    sem_d  = np.nanstd(dprimes)
    return mean_d, sem_d

def bootstrap_rates_ci(run_data, isi_value, n_boot=200):
    """
    Bootstrap mean ± SEM for hit and FA rates at a given ISI.
    Uses the ROC-based optimal criterion (closest point to (0,1)).
    """
    hits_raw = run_data["isi_hit_dists"].get(isi_value, [])
    if not hits_raw:
        return np.nan, np.nan, np.nan, np.nan

    hits = np.array([d for (d, t) in hits_raw], float)
    fas  = np.asarray(run_data["fas"], float)
    if len(hits) < 3 or len(fas) < 3:
        return np.nan, np.nan, np.nan, np.nan

    score_type = run_data.get("score_type", "distance")
    hit_rates, fa_rates = [], []

    for _ in range(n_boot):
        h_bs = resample(hits, replace=True)
        f_bs = resample(fas, replace=True)
        y_true = np.concatenate([np.ones(len(h_bs)), np.zeros(len(f_bs))])
        scores = np.concatenate([h_bs, f_bs])
        if score_type == "distance":
            scores = -scores
        fpr, tpr, _ = roc_curve(y_true, scores)
        dists = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
        i_best = np.argmin(dists)
        hit_rates.append(tpr[i_best])
        fa_rates.append(fpr[i_best])

    hit_rates = np.array(hit_rates)
    fa_rates  = np.array(fa_rates)
    return hit_rates.mean(), hit_rates.std(), fa_rates.mean(), fa_rates.std()



def compute_scaling_vs_human(runs, noise_levels, y_ref):
    """
    Fit human d' = α * model d' + β for each noise level.
    Returns per-noise-level dict with alpha, beta, r2.
    """
    scaling_results = {}
    
    for nv in noise_levels:
        run_data = runs[nv]
        model_isi, model_d = compute_model_dprime_curve(run_data)
    
        # linear fit: model → human
        reg = LinearRegression().fit(model_d.reshape(-1,1), y_ref)
        alpha, beta = reg.coef_[0], reg.intercept_
        r2 = reg.score(model_d.reshape(-1,1), y_ref)
    
        scaling_results[nv] = dict(alpha=alpha, beta=beta, r2=r2)
        
    return scaling_results

def convert_human_to_model_struct(main_exp):
    """
    Convert a participant dataframe into model-compatible format.
    
    main_exp : pandas.DataFrame
        Must include columns ['stimulus', 'repeat', 'isi', 'response'].
    
    Returns a dict matching the model-run output format.
    """
    # Convert types
    is_repeat = np.array(main_exp['repeat'] == 'true', dtype=bool)
    isi = np.array(main_exp['isi'], dtype=float)
    response = np.array(main_exp['response'], dtype=int)

    # --- Separate hit / false alarm "scores" ---
    # We'll use responses as "scores" (higher = more confident yes)
    hits = response[is_repeat]
    fas  = response[~is_repeat]

    # --- Build isi_hit_dists like the model structure ---
    isi_hit_dists = defaultdict(list)
    for i, (rep, isi_val, resp) in enumerate(zip(is_repeat, isi, response)):
        if rep and isi_val > -1:  # only valid repeats
            isi_hit_dists[int(isi_val)].append((float(resp), i))

    # --- Optional: FA-by-time list ---
    T_max = len(main_exp)
    fa_by_t = [[] for _ in range(T_max)]
    for t, (rep, resp) in enumerate(zip(is_repeat, response)):
        if not rep:
            fa_by_t[t].append(float(resp))

    return {
        "hits": np.asarray(hits, float),
        "fas": np.asarray(fas, float),
        "isi_hit_dists": isi_hit_dists,
        "fa_by_t": fa_by_t,
        "T_max": T_max,
        "score_type": "likelihood",
        "noise_mode": "human",
    }

def auroc_to_dprime(auroc):
    """Convert AUROC to d′ via z-transform rule."""
    auroc = np.clip(auroc, 1e-6, 1 - 1e-6)
    return np.sqrt(2) * norm.ppf(auroc)

def rocs_across_noise(
    noise_levels,
    *,
    runner,
    X0, name_to_idx, experiment_list,
    DistanceMemoryModel=None, zscore_projector=None, DEVICE="cpu", **kwargs
):
    """Run multiple simulations across noise levels."""
    curves, runs = {}, {}
    for nv in noise_levels:
        run = runner(
            nv,
            X0=X0, name_to_idx=name_to_idx, experiment_list=experiment_list,
            DistanceMemoryModel=DistanceMemoryModel, zscore_projector=zscore_projector,
            DEVICE=DEVICE, **kwargs
        )
        runs[nv] = run
        score_type = run.get("score_type", "distance")
        res = roc_from_arrays(run["hits"], run["fas"], score_type=score_type)
        if res is not None:
            curves[nv] = res
    return curves, runs


def roc_for_isi(run_data, isi_value):
    """Compute ROC for a specific ISI value."""
    hits_raw = run_data["isi_hit_dists"].get(isi_value, [])
    if not hits_raw:
        return None
    hits = np.array([d for (d, t) in hits_raw], float)
    fas = np.asarray(run_data["fas"], float)
    return roc_from_arrays(hits, fas, score_type=run_data.get("score_type", "distance"))


def roc_for_second_half(run_data):
    """Compute ROC using only the second half of trials."""
    T_half = run_data["T_max"] // 2
    hits, fas = [], []

    for lst in run_data["isi_hit_dists"].values():
        hits.extend([d for (d, t) in lst if t > T_half])
    hits = np.asarray(hits, float)

    for t_idx, bucket in enumerate(run_data["fa_by_t"], start=1):
        if t_idx > T_half:
            fas.extend(bucket)
    fas = np.asarray(fas, float)

    return roc_from_arrays(hits, fas, score_type=run_data.get("score_type", "distance"))


def compute_rates_by_isi_optimal(run_data):
    """
    Compute hit/FA rates by ISI using the ROC-based optimal criterion
    (threshold minimizing distance to (0,1)).
    """
    isi_hit_dists = run_data["isi_hit_dists"]
    fa_by_t = run_data["fa_by_t"]
    score_type = run_data["score_type"]

    isis = sorted(isi_hit_dists.keys())
    hit_rates, fa_rates = [], []

    for isi in isis:
        hit_scores = np.asarray([s for (s, _) in isi_hit_dists[isi]], float)
        fa_scores = np.asarray(fa_by_t[isi - 1], float) if isi - 1 < len(fa_by_t) else np.array([])

        if hit_scores.size == 0 or fa_scores.size == 0:
            hit_rates.append(np.nan)
            fa_rates.append(np.nan)
            continue

        y_true = np.concatenate([np.ones(len(hit_scores)), np.zeros(len(fa_scores))])
        scores = np.concatenate([hit_scores, fa_scores])
        if score_type == "distance":
            scores = -scores

        fpr, tpr, _ = roc_curve(y_true, scores)
        dists = np.sqrt((fpr - 0)**2 + (tpr - 1)**2)
        i_best = np.argmin(dists)
        hit_rates.append(tpr[i_best])
        fa_rates.append(fpr[i_best])

    return isis, np.array(hit_rates), np.array(fa_rates)