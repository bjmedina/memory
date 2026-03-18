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
sys.path.append('/om2/user/jmhicks/projects/TextureStreaming/code/')
sys.path.append('/om2/user/bjmedina/auditory-memory/memory/utls/')
sys.path.append('/om2/user/bjmedina/auditory-memory/memory/src/model/')
sys.path.append("/om2/user/bjmedina/auditory-memory/memory/")

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


def refine_search_local(
    base_results,
    n_refine,
    param_bounds,
    *,
    noise_mode,
    top_k=10,
    jitter_frac=0.01,
    rng=None
):
    rng = np.random.default_rng(rng)

    # Sort by coarse fit (lower is better)
    base_results = sorted(base_results, key=lambda r: r["nmse"])

    # Keep only top-K elites
    elites = base_results[:top_k]

    refined_params = []

    for base in elites:
        p0 = base["params"]

        for _ in range(n_refine):
            p = {}
            for k, (lo, hi) in param_bounds.items():
                span = hi - lo
                val = p0[k] + rng.normal(scale=jitter_frac * span)
                p[k] = np.clip(val, lo, hi)

            if noise_mode == "two-regime":
                if p["sigma0"] <= p["sigma1"]:
                    continue

            refined_params.append(p)

    return refined_params

def evaluate_params(
    params_list,
    *,
    noise_mode,
    metric,
    X0,
    name_to_idx,
    experiment_list,
    isis,
    human_curve,
    layer,
    encoder_name,
    hr_task_name,
    debug=False
):
    results = []

    for params in params_list:
        params = dict(params)  # defensive copy

        params.update({
            "noise_mode": noise_mode,
            "metric": metric,
            "layer": layer,
            "encoder": encoder_name,
            "stimulus_set": hr_task_name,
        })

        run_out = run_experiment_scores(
            sigma0=params["sigma0"],
            sigma1=params.get("sigma1", None),
            t_step=params.get("t_step", None),
            rate=params.get("rate", None),
            noise_mode=noise_mode,
            metric=metric,
            X0=X0,
            name_to_idx=name_to_idx,
            experiment_list=experiment_list,
            debug=debug,
        )
        
        model_dp = compute_model_dprime_for_run(run_out, isis)

        results.append({
            "params": params,
            "results": run_out,
            "model_dp": model_dp,
            "nmse": compute_nmse(model_dp, human_curve),
            "nmse_no_0": compute_nmse(model_dp, human_curve, start_idx=1),
            "mse": compute_mse(model_dp, human_curve),
            "mse_no_0": compute_mse(model_dp, human_curve, start_idx=1),
        })

    return results

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

    version = "v10_three-regime"
    save_figs = (
        f"{results_root}/"
        f"figures/"
        f"figures_{version}/"
        f"{task_name}/"
        f"{encoder_type}/"
        f"{metric}/"
        f"{noise_tag}/"
    )

    save_fits = f"{results_root}/fits/fits_{version}_best"
    save_fits_all = f"{results_root}/fits/fits_{version}_all"

    
    print(save_figs, save_fits)
    ensure_dir(save_figs)
    
    ensure_dir(save_fits)
    ensure_dir(save_fits_all)

    
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
    
    d50 = 1 #dont mind this
    
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

    # ---------------------------
    # STAGE A: broad search for sigma0
    # ---------------------------
    opt_results_stage1 = random_search_gridlike(
        n_samples=100,   # larger here
        param_bounds=param_bounds,
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
        subsample=16
    )
    
    
    best_mse_isizero = min(r["mse_per_isi"][0] for r in opt_results_stage1)
    
    tol = 1.02  # or absolute epsilon if you prefer
    good_sigma0 = sorted({
        r["params"]["sigma0"]
        for r in opt_results_stage1
        if r["mse_per_isi"][0] <= best_mse_isizero * tol
    })

    # ---------------------------
    # STAGE B: search for sigma1
    # ---------------------------
    opt_results_stage2 = []
    
    for sigma0 in good_sigma0:
        param_bounds_cond = {
            "sigma0": (sigma0*0.99, sigma0*1.01),  # FIX sigma0
            "sigma1": (
                noise_cfg["sigma1_min"] * d50,
                noise_cfg["sigma1_max"] * d50,
            ),
            "sigma2": (
                noise_cfg["sigma2_min"] * d50,
                noise_cfg["sigma2_max"] * d50,
            ),
    
            "t_step": (noise_cfg["t_step"], noise_cfg["t_step"]),
        }
    
        res = random_search_gridlike(
            n_samples=40,   # focused search
            param_bounds=param_bounds_cond,
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
            subsample=32
        )
    
        opt_results_stage2.extend(res)

    
    best_mse_early = min(r["mse_early_no_zero"] for r in opt_results_stage2)
    
    good_sigma1 = sorted({
        r["params"]["sigma1"]
        for r in opt_results_stage2
        if r["mse_early_no_zero"] <= best_mse_early * tol
    })
    
    opt_results_stage3 = []
    # ---------------------------
    # STAGE C: search for sigma2
    # ---------------------------
    
    for sigma1 in good_sigma1:
        param_bounds_cond = {
            "sigma0": (sigma0, sigma0),  # FIX sigma0
            "sigma1": (
                sigma1*0.99,
                sigma1*1.01,
            ),
            "sigma2": (
                noise_cfg["sigma2_min"] * d50,
                noise_cfg["sigma2_max"] * d50,
            ),
    
            "t_step": (noise_cfg["t_step"], noise_cfg["t_step"]),
        }
    
        res = random_search_gridlike(
            n_samples=40,   # focused search
            param_bounds=param_bounds_cond,
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
            subsample=32
        )
    
        opt_results_stage3.extend(res)


    best_fits = generate_and_plot_model_decay_summary_v2(
        opt_results_stage3,
        human_curve,
        isis,
        savedir=save_figs,
        max_rows=1,
        verbose=True, 
        hr_task_name=hr_task_name, 
        encoder_name=encoder_name
    )

    summary_list = save_best_models(best_fits, save_fits_all, prefix=f"{task_name}-{encoder_name}")
    
    top_results = opt_results_stage3

    #plot_best_model_histograms(all_fits, isis, save_figs)

    # refined_params = refine_search_local(
    #     base_results=opt_results,
    #     top_k=5,
    #     n_refine=10,
    #     jitter_frac=0.1*d50,
    #     param_bounds=param_bounds,
    #     noise_mode=noise_mode,
    # )

    # # 3) Run full experiments on refined params
    # fine_results = evaluate_params(
    #     refined_params,
    #     noise_mode=noise_mode,
    #     metric=metric,
    #     X0=X,
    #     name_to_idx=name_to_idx,
    #     experiment_list=exp_list[model_cfg["experiment"]["n_seqs"]:model_cfg["experiment"]["n_seqs"]+8*3] + exp_list[model_cfg["experiment"]["n_seqs"]:model_cfg["experiment"]["n_seqs"]+8*3],
    #     isis=isis, 
    #     human_curve=human_curve,
    #     layer=layer,
    #     encoder_name=encoder_name,
    #     hr_task_name=hr_task_name,
    # )
    
    # best_fits = generate_and_plot_model_decay_summary_v2(
    #     fine_results,
    #     human_curve,
    #     isis,
    #     savedir=save_figs,
    #     max_rows=1,
    #     verbose=True, hr_task_name=hr_task_name, encoder_name=encoder_name)
    
    # summary_list = save_best_models(best_fits, save_fits, prefix=f"{task_name}-{encoder_name}")
    
    # plot_best_model_histograms(best_fits, isis, save_figs)
