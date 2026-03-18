# ===================== Imports =====================
import argparse, sys, os, glob, json, math, datetime, torch, yaml
import matplotlib.pyplot as plt, numpy as np, pandas as pd

from collections import defaultdict
from scipy.stats import norm, pearsonr

from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from pathlib import Path
from scipy.spatial.distance import pdist




# project-specific paths
sys.path.append('/orcd/data/jhm/001/om2/jmhicks/projects/TextureStreaming/code/')
sys.path.append('/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/utls/')
sys.path.append('/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/src/model/')
sys.path.append("/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/")

from chexture_choolbox.auditorytexture.texture_model import TextureModel
from chexture_choolbox.auditorytexture.helpers import FlattenStats
from texture_prior.params import model_params, statistics_set, texture_dataset
from texture_prior.utils import path

from utls.encoders import *
from utls.plotting import ensure_dir
from utls.loading import load_results_with_exclusion_2, move_sequences_to_used, load_results_with_exclusion_no_dropping
from utls.runners import run_experiment_scores
from utls.runners_v2 import (
    run_experiment_grid,
    run_experiment_scores,
    run_experiment_scores_itemwise,
    run_experiment_itemwise_hits_fas,
    make_noise_schedule
)

from utls.analysis_helpers import rocs_across_noise, convert_human_to_model_struct, compute_scaling_vs_human, convert_human_to_model_struct
from utls.analysis_helpers import auroc_to_dprime, compute_model_dprime_curve
from utls.analysis_helpers import roc_for_isi, auroc_to_dprime
from utls.plotting import plot_across_noise, plot_noise_overlays
from utls.io_utils import make_model_save_dir, save_all_figures, save_single_figure, save_runs_summary
from utls.roc_utils import roc_from_arrays 
from utls.runners_utils import *


def load_config(cfg_path=None):
    """
    Load YAML config.
    Priority:
      1. cfg_path argument (if provided)
      2. sys.argv[1]
    """
    if cfg_path is None:
        if len(sys.argv) != 2:
            raise RuntimeError("Usage: python main.py path/to/run.yaml")
        cfg_path = sys.argv[1]

    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    with open(cfg_path, "r") as f:
        return yaml.safe_load(f), cfg_path

import itertools
import numpy as np

def log_grid(lo, hi, n):
    """Deterministic log-spaced grid between lo and hi (inclusive-ish)."""
    lo = max(float(lo), 1e-12)
    hi = max(float(hi), lo * 1.0001)
    if n <= 1 or lo == hi:
        return [lo]
    return list(np.exp(np.linspace(np.log(lo), np.log(hi), n)))

def is_valid(params, noise_mode):
    """Regime constraints (edit if your ordering assumption differs)."""
    if noise_mode == "two-regime":
        # enforce sigma0 > sigma1
        return params["sigma0"] >= params["sigma1"]
    if noise_mode == "three-regime":
        # enforce sigma0 > sigma1 > sigma2
        return params["sigma0"] >= params["sigma1"] > params["sigma2"]
    return True

def generate_candidates(param_bounds, *, free_keys, fixed_params=None, n_per_dim=5, noise_mode="three-regime"):
    """
    Build deterministic param dicts.

    - free_keys: list of keys to grid over
    - fixed_params: dict of fixed values (optional)
    - keys not in free_keys are set to:
        * fixed_params[k] if provided, else
        * lo (must equal hi) from param_bounds
    """
    fixed_params = {} if fixed_params is None else dict(fixed_params)

    # build per-key value lists
    values_by_key = {}
    for k, (lo, hi) in param_bounds.items():
        if k in fixed_params:
            values_by_key[k] = [float(fixed_params[k])]
        elif k in free_keys:
            values_by_key[k] = log_grid(lo, hi, n_per_dim)
        else:
            # held fixed by bounds; if not fixed, require lo==hi
            if lo != hi:
                raise ValueError(f"Key '{k}' not in free_keys and not fixed_params, but bounds vary: {(lo,hi)}")
            values_by_key[k] = [float(lo)]

    keys = list(values_by_key.keys())
    grids = [values_by_key[k] for k in keys]

    candidates = []
    for vals in itertools.product(*grids):
        p = {k: v for k, v in zip(keys, vals)}
        if is_valid(p, noise_mode):
            candidates.append(p)

    return candidates

def median_pairwise_distance(X, metric="euclidean", n_samples=500, seed=0):
    """
    Estimate the median pairwise distance under a given metric.

    Args:
        X (np.ndarray): shape (N, D). Must already be preprocessed
                        appropriately for the metric
                        (e.g., L2-normalized for cosine).
        metric (str): distance metric for scipy.spatial.distance.pdist
                      ("cosine", "euclidean", "mahalanobis", etc.).
        n_samples (int): number of items to subsample for efficiency.
        seed (int): RNG seed.

    Returns:
        float: median pairwise distance.
    """
    rng = np.random.default_rng(seed)
    N = X.shape[0]

    idx = rng.choice(N, size=min(n_samples, N), replace=False)
    Xs = X[idx]

    dists = pdist(Xs, metric=metric)
    d50 = np.median(dists)

    return float(d50)

tasks = {
    0: "ind-nature",
    1: "global-music",
    2: "atexts",
}

if __name__ == "__main__":

    model_cfg, model_cfg_path = load_config()

    print("Running config:")
    print("  YAML:", model_cfg_path)
    print("  Encoder:", model_cfg["representation"]["type"])
    print("  Metric:", model_cfg["metric"])
    print("  Noise:", model_cfg["noise_model"]["name"])

    # ---------------------------
    # SETTINGS (from YAML)
    # ---------------------------
    
    # ---- experiment ----
    exp_cfg = model_cfg["experiment"]
    
    which_task = exp_cfg["which_task"]
    is_multi   = exp_cfg["is_multi"]
    which_isi  = exp_cfg.get("which_isi", None)
    
    if is_multi:
        isis = [0, 1, 2, 4, 8, 16, 32, 64]
    else:
        assert which_isi is not None, "which_isi required if not multi-ISI"
        isis = [0, which_isi]
    
    # ---- metric ----
    metric = model_cfg["metric"]
    
    # ---- noise model ----
    noise_cfg = model_cfg["noise_model"]
    noise_mode = noise_cfg["name"]
    if "t_step" in noise_cfg:
        t_step = model_cfg["noise_model"]["t_step"]
    
    # ---- representation ----
    repr_cfg = model_cfg["representation"]
    time_avg = repr_cfg["time_avg"]
    
    encoder_type = repr_cfg["type"]
    layer   = repr_cfg.get("layer", None)
    PC_DIMS = repr_cfg.get("pc_dims", None)
    
    # ---------------------------
    # 1. Load data
    # ---------------------------
    exp_list, all_files, name_to_idx, human_runs, task_name, hr_task_name = load_experiment_data(
        which_task, which_isi, is_multi)
    
    # ---------------------------
    # 2. Human curve
    # ---------------------------
    human_curve = compute_human_curve(human_runs, is_multi, which_isi)
    results_root = model_cfg["results_root"]
    tag = model_cfg.get("tag", "untagged")
    
    if noise_mode == "two-regime":
        assert "t_step" in noise_cfg, "two-regime requires t_step"
        t_step = noise_cfg["t_step"]
        noise_tag = f"{noise_mode}_t{t_step}"
    else:
        noise_tag = noise_mode

    if time_ag
    version = "v12_three-regime"
    save_figs = (
        f"{results_root}/"
        f"figures/"
        f"figures_{version}/"
        f"{task_name}/"
        f"{encoder_type}/"
        f"{metric}/"
        f"{noise_tag}/"
    )

    save_fits     = f"{results_root}/fits/fits_{version}_best"
    save_fits_all = f"{results_root}/fits/fits_{version}_all"
    save_fits_int = f"{results_root}/fits/fits_{version}_intermediate"

    
    print(save_figs, save_fits)
    
    ensure_dir(save_figs)
    ensure_dir(save_fits)
    ensure_dir(save_fits_all)
    ensure_dir(save_fits_int)


    
    encoder_type = repr_cfg["type"]     # e.g. "kell2018"
    layer        = repr_cfg.get("layer")
    pc_dims      = repr_cfg.get("pc_dims")

    NN_ENCODERS = {"kell2018", "resnet50"}

    encoder_task = (
        "word_speaker_audioset"
        if encoder_type in NN_ENCODERS
        else "audioset"
    )
        
    encoder_cfg = dict(
        encoder_type=encoder_type,      # e.g. "resnet50"
        model_name=encoder_type,        # same by design
        task=encoder_task,
        statistics_dict=statistics_set.statistics,
        model_params=model_params,
        sr=20000,
        duration=2.0,
        rms_level=0.05,
        device="cuda",
    )
    
    # ---- representation-specific fields ----
    if encoder_type in ("kell2018", "resnet50"):
        encoder_cfg["layer"] = layer
        assert layer is not None, f"layer required for {encoder_type}"
    
    if encoder_type == "texture_pca":
        encoder_cfg["pc_dims"] = pc_dims
        assert pc_dims is not None, "pc_dims required for texture encoder"
    
    encoder_name = make_encoder_name(encoder_cfg)
    print("Encoder name:", encoder_name)
    
    encoder = build_encoder(encoder_cfg)
    X = encode_stimuli(encoder, all_files)

    # X is torch tensor at this point
    X_np = X.detach().cpu().numpy()
    
    # d50 = median_pairwise_distance(
    #     X_np,
    #     metric="cosine",      # e.g. "cosine", "euclidean", "mahalanobis"
    #     n_samples=500,
    #     seed=0,
    # ) 

    d50 = 1
    
    noise_cfg  = model_cfg["noise_model"]
    noise_mode = noise_cfg["name"]
    
    # ----------------------------------
    # Noise parameter bounds (from YAML)
    # ----------------------------------
    param_bounds = {
        "sigma0": (noise_cfg["sigma0_min"]* d50, noise_cfg["sigma0_max"]* d50),
    }
    
    if noise_mode == "two-regime":
        param_bounds["sigma1"] = (
            noise_cfg["sigma1_min"]* d50,
            noise_cfg["sigma1_max"]* d50,
        )
        param_bounds["t_step"] = (
            noise_cfg["t_step"],
            noise_cfg["t_step"],
        )
        
    if noise_mode == "three-regime":
        param_bounds["sigma1"] = (
            noise_cfg["sigma1_min"]* d50,
            noise_cfg["sigma1_max"]* d50,
        )

        param_bounds["sigma2"] = (
            noise_cfg["sigma2_min"]* d50,
            noise_cfg["sigma2_max"]* d50,
        )
        param_bounds["t_step"] = (
            noise_cfg["t_step"],
            noise_cfg["t_step"],
        )

    paramA = dict(param_bounds)
    
    def log_mid(lo, hi):
        lo, hi = max(lo, 1e-12), max(hi, 1e-12)
        return float(np.exp(0.5 * (np.log(lo) + np.log(hi))))
    
    print("median distance:", d50)
    
    # ---- IMPORTANT: actually use these bounds ----
    sigma0_min = 1.0
    sigma0_max = 5.0 
    paramA["sigma0"] = (sigma0_min, sigma0_max)
    
    assert "sigma1" in param_bounds and "sigma2" in param_bounds, "param_bounds missing sigma1/sigma2"
    
    s1_fix = log_mid(*param_bounds["sigma1"])
    s2_fix = log_mid(*param_bounds["sigma2"])
    paramA["sigma1"] = (s1_fix, s1_fix)
    paramA["sigma2"] = (s2_fix, s2_fix)
    
    cands = generate_candidates(
        paramA,
        free_keys=["sigma0"],
        fixed_params={"sigma1": s1_fix, "sigma2": s2_fix},
        n_per_dim=20,                # 20 evaluations
        noise_mode=noise_mode
    )
    
    print(f"there are {len(cands)} candidates") 
    resA = []
    for c in cands:
    
        curr_pb = dict(paramA)
        curr_pb["sigma1"] = (c["sigma1"], c["sigma1"])
        curr_pb["sigma2"] = (c["sigma2"], c["sigma2"])
        curr_pb["t_step"] = (c["t_step"], c["t_step"])
    
        resA_single = random_search_gridlike(
            n_samples=1,
            param_bounds=curr_pb,
            noise_mode=noise_mode,
            metric=metric,
            X0=X,
            name_to_idx=name_to_idx,
            experiment_list=exp_list,
            isis=isis,
            human_curve=human_curve,
            layer=encoder_name,
            encoder_name=encoder_name,
            hr_task_name=hr_task_name,
            debug=False,
            subsample=8,
            random_draw=False
        )[0]
    
        resA.append(resA_single)
    
    tol0_band = 0.1
    bestA = min(resA, key=lambda r: float(r["mse_per_isi"][0]))
    sigma0_star = bestA["params"]["sigma0"]
    print("Stage A best sigma0:", sigma0_star, "mse_isi0:", bestA["mse_per_isi"][0])


    best_fits = generate_and_plot_model_decay_summary_v5(
        resA,
        human_curve,
        isis,
        metric_name="mse_per_isi",
        isi_indices=[0],
        savedir=save_figs,
        max_rows=1,
        hr_task_name=hr_task_name, 
        encoder_name=encoder_name
    )

    summary_list = save_best_models(best_fits, save_fits_all, prefix=f"{task_name}-{encoder_name}-isizero")

    tol0_band = 0.05
    
    # ---- Stage B ----
    paramB = dict(param_bounds)
    paramB["sigma0"] = (sigma0_star*(1-tol0_band), sigma0_star*(1+tol0_band))
    
    print(paramB["sigma0"])
    
    paramB["sigma1"] = (0.01, sigma0_star*(1-tol0_band))
    paramB["sigma2"] = (0.0001, sigma0_star*(1-tol0_band))
    
    cands = generate_candidates(
        paramB,
        free_keys=["sigma0", "sigma1", "sigma2"],
        fixed_params=None,
        n_per_dim=5,                 # 7x7 = 49 evaluations
        noise_mode=noise_mode
    )
    
    resB = []
    for j in range(len(cands)):
        c = cands[j]
        print(f"trying out {c}")
    
        curr_pb = dict(paramB)
        curr_pb["sigma0"] = (c["sigma0"], c["sigma0"])
        curr_pb["sigma1"] = (c["sigma1"], c["sigma1"])
        curr_pb["sigma2"] = (c["sigma2"], c["sigma2"])
        curr_pb["t_step"] = (c["t_step"], c["t_step"])
    
        resB_single = random_search_gridlike(
            n_samples=1,
            param_bounds=curr_pb,
            noise_mode=noise_mode,
            metric=metric,
            X0=X,
            name_to_idx=name_to_idx,
            experiment_list=exp_list,
            isis=isis,
            human_curve=human_curve,
            layer=encoder_name,
            encoder_name=encoder_name,
            hr_task_name=hr_task_name,
            debug=False,
            subsample=8, 
            random_draw=False
        )[0]
    
        resB.append(resB_single)
    
    def obj_ge2(r):
        v = np.asarray(r["mse_per_isi"], float)
        return float(np.nanmean(v[1:]))
    
    bestB = min(resB, key=obj_ge2)
    print("Stage B best params:", bestB["params"])
    print("Stage B mse_per_isi:", bestB["mse_per_isi"])
    print("Stage B obj (ISI>=1 mean):", obj_ge2(bestB))

    best_fits = generate_and_plot_model_decay_summary_v5(
        resB,
        human_curve,
        isis,
        metric_name="mse_per_isi",
        isi_indices=[1,2,3,4,5,6,7],
        savedir=save_figs,
        max_rows=1,
        hr_task_name=hr_task_name, 
        encoder_name=encoder_name)

    summary_list = save_best_models(best_fits, save_fits_int, prefix=f"{task_name}-{encoder_name}")


    # bestB is your Stage B best result
    best = bestB["params"]
    
    def mult_band(x, frac):
        return (x * (1 - frac), x * (1 + frac))
    
    # refine bounds (tight)
    refine_bounds = dict(param_bounds)
    refine_bounds["sigma0"] = mult_band(best["sigma0"], 0.10)  # ±10%
    refine_bounds["sigma1"] = mult_band(best["sigma1"], 0.10)  # ±20%
    refine_bounds["sigma2"] = mult_band(best["sigma2"], 0.10)  # ±20%
    refine_bounds["t_step"] = (noise_cfg["t_step"], noise_cfg["t_step"])  # fixed
    
    # clip to global bounds so we don't exceed YAML ranges
    def clip_bound(b, global_b):
        lo, hi = b
        glo, ghi = global_b
        return (max(lo, glo), min(hi, ghi))
    
    refine_bounds["sigma0"] = clip_bound(refine_bounds["sigma0"], param_bounds["sigma0"])
    refine_bounds["sigma1"] = clip_bound(refine_bounds["sigma1"], param_bounds["sigma1"])
    refine_bounds["sigma2"] = clip_bound(refine_bounds["sigma2"], param_bounds["sigma2"])
    
    # generate a small deterministic grid: 5^3 = 125
    cands = generate_candidates(
        refine_bounds,
        free_keys=["sigma0", "sigma1", "sigma2"],
        fixed_params={"t_step": noise_cfg["t_step"]},  # optional if bounds already fixed
        n_per_dim=5,
        noise_mode=noise_mode
    )
    
    refined = []
    for k in range(len(cands)):
        p = cands[k]
        print(f"refined search candidate: {p}")
        pb = dict(refine_bounds)
        pb["sigma0"] = (p["sigma0"], p["sigma0"])
        pb["sigma1"] = (p["sigma1"], p["sigma1"])
        pb["sigma2"] = (p["sigma2"], p["sigma2"])
    
        r = random_search_gridlike(
            n_samples=1,
            param_bounds=pb,
            noise_mode=noise_mode,
            metric=metric,
            X0=X,
            name_to_idx=name_to_idx,
            experiment_list=exp_list,
            isis=isis,
            human_curve=human_curve,
            layer=encoder_name,
            encoder_name=encoder_name,
            hr_task_name=hr_task_name,
            debug=False,
            subsample=8,
            random_draw=False
        )[0]
        refined.append(r)
    
    best_refined = min(refined, key=obj_ge2)
    print("REFINED best params:", best_refined["params"])
    print("REFINED mse_per_isi:", best_refined["mse_per_isi"])
    print("REFINED obj:", obj_ge2(best_refined))

    best_fits = generate_and_plot_model_decay_summary_v5(
        refined,
        human_curve,
        isis,
        metric_name="mse_per_isi",
        isi_indices=[1,2,3,4,5,6,7],
        savedir=save_figs,
        max_rows=1,
        hr_task_name=hr_task_name, 
        encoder_name=encoder_name)
    
    summary_list_best = save_best_models(best_fits, save_fits, prefix=f"{task_name}-{encoder_name}")
