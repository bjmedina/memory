# ===================== Imports =====================
import argparse, sys, os, glob, json, math, random, datetime, torch, re
import matplotlib.pyplot as plt, numpy as np, pandas as pd

from collections import defaultdict
from scipy.stats import norm, pearsonr

from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

from matplotlib.gridspec import GridSpec
from pathlib import Path


# project-specific paths
sys.path.append('/om2/user/jmhicks/projects/TextureStreaming/code/')
sys.path.append('../utls/')
sys.path.append('../src/model/')
sys.path.append("/om2/user/bjmedina/auditory-memory/memory/")

from chexture_choolbox.auditorytexture.texture_model import TextureModel
from chexture_choolbox.auditorytexture.helpers import FlattenStats
from texture_prior.params import model_params, statistics_set, texture_dataset
from texture_prior.utils import path

from utls.encoders import *
from utls.plotting import ensure_dir
from utls.loading import load_results_with_exclusion_2, move_sequences_to_used, load_results_with_exclusion_no_dropping, refresh_unused_batch
#from utls.runners import run_experiment_scores
from utls.runners_v2 import (
    run_experiment_grid,
    run_experiment_scores,
    run_experiment_scores_v2,
    run_experiment_scores_itemwise,
    run_experiment_scores_itemwise_v2,
    run_experiment_itemwise_hits_fas,
    make_noise_schedule
)

from utls.analysis_helpers import rocs_across_noise, convert_human_to_model_struct, compute_scaling_vs_human, convert_human_to_model_struct
from utls.analysis_helpers import auroc_to_dprime, compute_model_dprime_curve
from utls.analysis_helpers import roc_for_isi, auroc_to_dprime
from utls.plotting import plot_across_noise, plot_noise_overlays
from utls.io_utils import make_model_save_dir, save_all_figures, save_single_figure, save_runs_summary
from utls.roc_utils import roc_from_arrays 


# ===============================================================
# A. ——— Experiment Loading
# ===============================================================

def cosine_similarity_dp(model_dp, human_dp, *, start_idx=1):
    x = np.asarray(model_dp[start_idx:], float)
    y = np.asarray(human_dp[start_idx:], float)

    mask = ~np.isnan(x) & ~np.isnan(y)
    x, y = x[mask], y[mask]

    if len(x) == 0:
        return np.nan

    return float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))

def load_experiment_data(which_task, which_isi, is_multi, old=False):
    """
    Returns:
        experiment_list (list[list[str]])
        all_files       (list[str])
        name_to_idx     (dict)
        human_runs      (list)
    """

    batch_size = 8
    active_batch_size = batch_size*18
    if not is_multi:
        tasks = ["ind-nature-len120", "global-music-len120",
                 "atexts-len120", "nhs-region-len120"]
        base_path = f"/mindhive/mcdermott/www/mturk_stimuli/bjmedina/{{}}/sequences/isi_{which_isi}/len120/"
    else:
        tasks = ["env-sounds", "glob-music", "atexts", "nhs-region-len120"]
        base_path = "/mindhive/mcdermott/www/mturk_stimuli/bjmedina/{}/sequences/len120_multi/"

    task_name = tasks[which_task]

    # map tasks → set names
    seqs_paths = {
        tasks[0]: "mem_exp_ind-nature_2025",
        tasks[1]: "global-music-2025-n_80",
        tasks[2]: "mem_exp_atexts_2025",
        tasks[3]: "nhs-region-n_80",
    }

    hr_task_names = { tasks[0]: "Industrial and Nature", 
                     tasks[1]: "Globalized Music",
                     tasks[2]: "Auditory Textures",
                     tasks[3]: " 'Natural History of Song' "}

    hr_task_name = hr_task_names[task_name]

    # load human data
    if not is_multi:
        exps, seqs, fnames, _, _, _ = load_results_with_exclusion_no_dropping(
            f"/mindhive/mcdermott/www/bjmedina/experiments/bolivia_2025/results/"
            f"isi_{which_isi}/{task_name}",
            min_dprime=2, min_trials=120, skip_len60=True,
            verbose=False, return_skipped=True)
    else:
        exps, seqs, fnames, _, _, _ = load_results_with_exclusion_no_dropping(
            f"/mindhive/mcdermott/www/bjmedina/experiments/{task_name}/results/"
            f"{task_name}/len120_multi",
            min_dprime=2, min_trials=120, skip_len60=True,
            verbose=False, return_skipped=True)

    # ---- group indices by sequence ----
    seq_to_indices = defaultdict(list)
    for i, s in enumerate(seqs):
        seq_to_indices[s].append(i)
    
    # ---- randomly keep one index per sequence ----
    keep = [random.choice(idxs) for idxs in seq_to_indices.values()]
    
    # ---- apply filter ----
    exps   = [exps[i]   for i in keep]
    seqs   = [seqs[i]   for i in keep]
    fnames = [fnames[i] for i in keep]

    if not old:

        current_batch = refresh_unused_batch(base_path.format(seqs_paths[task_name]), 8)
        
        # ---- extract seq numbers ----
        seqnums = [int(re.search(r"seq(\d+)", s).group(1)) for s in seqs]
        
        # ---- group seqnums by batch ----
        batch_to_seqnums = defaultdict(set)
        for n in seqnums:
            batch_id = (n - 1) // batch_size
            batch_to_seqnums[batch_id].add(n)
        
        # ---- determine which batches are complete ----
        complete_batches = {
            b for b, nums in batch_to_seqnums.items()
            if nums == set(range(b * batch_size + 1, b * batch_size + batch_size + 1))
        }
        
        # ---- filter everything in lockstep ----
        keep = [
            i for i, n in enumerate(seqnums)
            if (n - 1) // batch_size in complete_batches
        ]
        
        exps   = [exps[i] for i in keep]
        seqs   = [seqs[i] for i in keep]
        fnames = [fnames[i] for i in keep]

    # load stimulus sequences
    experiment_list = []
    seq_dir = base_path.format(seqs_paths[task_name])
    stim_base = "/".join(seq_dir.split("/")[:-3])

    for seq in seqs:
        with open(seq_dir + seq, "r") as f:
            data = json.load(f)
        stim_files = [stim_base + "/" + s for s in data["filenames_order"]]
        experiment_list.append(stim_files)

    # collapse all unique files
    all_files = sorted({fn for seq in experiment_list for fn in seq})
    name_to_idx = {fn: i for i, fn in enumerate(all_files)}

    # convert human runs
    human_runs = [convert_human_to_model_struct(e) for e in exps]

    return experiment_list, all_files, name_to_idx, human_runs, task_name, hr_task_name

def make_encoder_name(cfg):
    base = cfg.get("encoder_type", "unknown")

    # include layer if present
    layer = cfg.get("layer", None)
    if layer:
        base += f"-{layer}"

    # include PCA dims if present
    pc = cfg.get("pc_dims", None)
    if pc:
        base += f"-PC{pc}"

    return base

# ===============================================================
# B. ——— d′ Curves
# ===============================================================

def compute_human_curve(human_runs, is_multi, which_isi):
    if not is_multi:
        isis = [0, which_isi]
    else:
        isis = [0, 1, 2, 3, 4, 8, 16, 32, 64]

    dprimes = []
    for isi in isis:
        aucs = []
        for run in human_runs:
            res = roc_for_isi(run, isi)
            if res is not None:
                _, _, A = res
                aucs.append(auroc_to_dprime(A))
        dprimes.append(np.nanmean(aucs))

    return np.array([v for v in dprimes if not np.isnan(v)], float)

def compute_model_dprime_for_run(run_out, isis):

    isi_hit_dists = run_out.get("isi_hit_dists", None)
    fa_by_t = run_out.get("fa_by_t", None)
    T_max = run_out.get("T_max", None)
    score_type = run_out.get("score_type", "distance")

    if isi_hit_dists is None or fa_by_t is None:
        raise ValueError("run_out must include isi_hit_dists and fa_by_t.")

    # -------------------------
    # BUILD GLOBAL FA POOL ONCE
    # -------------------------
    all_fa_scores = []
    for t in range(T_max):
        all_fa_scores.extend(fa_by_t[t])
    all_fa_scores = np.asarray(all_fa_scores, float)

    dprime_vals = []

    for isi in isis:
        entries = isi_hit_dists.get(isi + 1, [])
        if len(entries) == 0:
            dprime_vals.append(np.nan)
            continue

        # hit scores at this ISI
        hit_scores = np.asarray([s for (s, _) in entries], float)

        if hit_scores.size == 0 or all_fa_scores.size == 0:
            dprime_vals.append(np.nan)
            continue

        # ROC / AUROC / d′ using GLOBAL FA
        fpr, tpr, auc = roc_from_arrays(
            hit_scores,
            all_fa_scores,
            score_type=score_type
        )

        dp = auroc_to_dprime(auc)
        dprime_vals.append(dp)

    return np.asarray(dprime_vals)

# ===============================================================
# C. ——— Encoding
# ===============================================================

def build_encoder(cfg):

    sys.path.append(f'/om2/user/bjmedina/models/cochdnn/model_directories/{cfg['model_name']}_{cfg['task']}/')
    print("LOADING FROM", f'/om2/user/bjmedina/models/cochdnn/model_directories/{cfg['model_name']}_{cfg['task']}/')
       
    etype = cfg["encoder_type"]

    if etype == "kell2018":
        return Kell2018Encoder(
            model_name=cfg["model_name"],
            layer=cfg["layer"],
            sr=cfg["sr"],
            rms_level=cfg["rms_level"],
            duration=cfg["duration"],
            time_avg=cfg["time_avg"],
            device=cfg["device"]
        )
    elif etype == "resnet50":
        return ResNet50Encoder(
            model_name=cfg["model_name"],
            layer=cfg["layer"],
            sr=cfg["sr"],
            rms_level=cfg["rms_level"],
            duration=cfg["duration"],
            time_avg=cfg["time_avg"],
            device=cfg["device"]
        )
    elif etype == 'texture_pca':
        return AudioTextureEncoderPCA(
            statistics_dict=cfg['statistics_dict'],
            pc_dims=cfg['pc_dims'],
            model_params=cfg['model_params'],
            sr=cfg.get('sr', 20000),
            rms_level=cfg.get('rms_level', 0.05),
            duration=cfg.get('duration', 2.0),
            device=cfg.get('device', 'cuda')
        )
    else:
        raise ValueError(f"Unsupported encoder type: {etype}")


def encode_stimuli(encoder, file_list):
    feats = []
    for filepath in file_list:
        out = encoder(filepath)
        if isinstance(out, dict):
            out = out["embedding"]
        feats.append(out.reshape(-1))
    return torch.stack(feats, dim=0)

# ===============================================================
# D. ——— Model Run + Grid Search
# ===============================================================

def run_model_grid(
    X0, name_to_idx, experiment_list,
    metric="cosine",
    noise_grid=None,
    rate_grid=None,
    mode_grid=None,
):
    param_grid = {
        "sigma0": noise_grid,
        "rate": rate_grid,
        "noise_mode": mode_grid,
        "metric": [metric],
    }

    return run_experiment_grid(
        X0=X0,
        name_to_idx=name_to_idx,
        experiment_list=experiment_list,
        param_grid=param_grid,
        fixed_params={},
        debug=True,
    )

def compute_nmse(model_dp, human_dp, start_idx=0, end_idx=-1):
    """Normalized MSE between model and human d′ curves."""
    mask = ~np.isnan(model_dp)
    if mask.sum() == 0:
        return np.inf
    mse = np.mean((model_dp[mask][start_idx:end_idx] - human_dp[mask][start_idx:end_idx])**2)
    var = np.var(human_dp[mask][start_idx:])
    return mse / (var + 1e-12)

def compute_mse(model_dp, human_dp, start_idx=0, end_idx=-1):
    """MSE between model and human d′ curves."""
    mask = ~np.isnan(model_dp)
    if mask.sum() == 0:
        return np.inf
    mse = np.mean((model_dp[mask][start_idx:end_idx] - human_dp[mask][start_idx:end_idx])**2)
    return mse

def evaluate_grid_results(grid_results, human_curve, isis):
    """
    Compute d′ curve and regime-specific errors for each model entry.

    Adds fields:
        rec["model_dp"]
        rec["mse_early"]
        rec["mse_late"]
        rec["mse_all"]
    """
    for rec in grid_results:
        model_dp = compute_model_dprime_for_run(rec["results"], isis)
        rec["model_dp"] = model_dp

        # ---- determine regime split ----
        params = rec["params"]
        if "t_step" in params:
            t_step = params["t_step"]
        else:
            t_step = np.inf

        early_end, late_start = isi_split_indices(isis, t_step)

        # ---- compute MSEs ----
        rec["mse_early"] = compute_mse(
            model_dp, human_curve,
            start_idx=0,
            end_idx=early_end
        )

        rec["mse_late"] = compute_mse(
            model_dp, human_curve,
            start_idx=late_start,
            end_idx=-1
        )

        rec["mse_all"] = compute_mse(model_dp, human_curve)

        # ---- keep old diagnostics if you want ----
        rec["cosine_sim"] = cosine_similarity_dp(model_dp, human_curve)

    return grid_results

def get_best_by_class(grid_results, best_key="nmse"):
    """
    Returns best model (lowest NMSE) within each (metric, noise_mode) class.
    """
    best = {}
    for rec in grid_results:
        params = rec["params"]
        key = (params["metric"], params["noise_mode"])
        
        current_val = rec[best_key]
        if key not in best or current_val < best[key][best_key]:
            best[key] = rec
    
    return best

def print_best_models(best_models, best_key="nmse"):
    print("\n==== Best Models by Class (metric × noise_mode) ====\n")
    for (metric, noise_mode), rec in best_models.items():
        current_val = rec[best_key]
        params = rec["params"]
        print(f"[{metric} | {noise_mode}]  {best_key}={current_val:.4f}")
        print("  Params:", params)
        print()

def isi_split_indices(isis, t_step):
    """
    Given ISIs and a t_step, return index ranges for early and late regimes.

    Returns:
        early_end_idx : int  (exclusive)
        late_start_idx : int (inclusive)
    """
    early_idxs = [i for i, isi in enumerate(isis) if isi < t_step]
    late_idxs  = [i for i, isi in enumerate(isis) if isi >= t_step]

    if len(early_idxs) == 0:
        early_end = 0
    else:
        early_end = max(early_idxs) + 1

    if len(late_idxs) == 0:
        late_start = len(isis)
    else:
        late_start = min(late_idxs)

    return early_end, late_start

def random_optimize(
    n_samples,
    param_bounds,
    noise_mode,
    metric,
    X0,
    name_to_idx,
    experiment_list,
    isis,
    human_curve,
    seed=123,
    best_key="nmse"
):
    """
    Random sampling optimization for your memory model.

    param_bounds: dict
        { "sigma0": (min,max), "rate": (min,max), "sigma1": (min,max), ... }

    Returns dict with best params + best NMSE.
    """
    rng = np.random.default_rng(seed)
    best_nmse = np.inf
    best_params = None

    keys = list(param_bounds.keys())

    for _ in range(n_samples):

        # Sample random parameter values
        params = {}
        for k in keys:
            lo, hi = param_bounds[k]
            params[k] = rng.uniform(lo, hi)

        # Build kwargs for the model call
        model_kwargs = dict(
            noise_mode=noise_mode,
            metric=metric,
            X0=X0,
            name_to_idx=name_to_idx,
            experiment_list=experiment_list,
        )
        model_kwargs.update(params)

        # run model
        run_out = run_experiment_scores(**model_kwargs)

        # compute model d'
        model_dp = compute_model_dprime_for_run(run_out, isis)

        # compute NMSE
        nmse = compute_nmse(model_dp, human_curve)

        if nmse < best_nmse:
            best_nmse = nmse
            best_params = params.copy()

    return best_params, best_nmse

    

def l2_distance(a, b):
    """Euclidean (L2) distance between two vectors."""
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.linalg.norm(a - b))

def l1_distance(a, b):
    """L1 (Manhattan) distance between two vectors."""
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.sum(np.abs(a - b)))

def mse_per_isi(model_dp, human_curve, *, isis=None):
    """
    Pointwise MSE per ISI (i.e., squared error at each ISI).

    Returns:
        per_isi: np.ndarray shape (n_isi,)
        per_isi_dict: dict mapping ISI -> mse_at_that_isi (if isis provided)
    """
    m = np.asarray(model_dp, dtype=float)
    h = np.asarray(human_curve, dtype=float)
    assert m.shape == h.shape, f"Shape mismatch: {m.shape} vs {h.shape}"

    per_isi = (m - h) ** 2  # single-point MSE

    # keep NaNs where either side is NaN
    bad = ~np.isfinite(m) | ~np.isfinite(h)
    if np.any(bad):
        per_isi = per_isi.copy()
        per_isi[bad] = np.nan

    per_isi_dict = None
    if isis is not None:
        isis = np.asarray(isis)
        assert isis.shape == per_isi.shape, f"isis shape mismatch: {isis.shape} vs {per_isi.shape}"
        per_isi_dict = {float(isi): float(v) if np.isfinite(v) else np.nan
                        for isi, v in zip(isis, per_isi)}

    return per_isi, per_isi_dict
    
####
def random_search_gridlike(
    n_samples,
    param_bounds,
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
    debug=False,
    subsample=8,
    random_draw=True,
    seed=0
):
    rng = np.random.default_rng(seed)
    all_results = []

    for i in range(n_samples):
        while True:
            params = {}
            for key, bound in param_bounds.items():
                lo, hi = bound
                if lo == hi:
                    params[key] = lo 
                else:
                    params[key] = np.exp(
                        np.random.uniform(np.log(lo), np.log(hi))
                    )
    
            if noise_mode == "two-regime":
                if params.get("sigma0", -np.inf) <= params.get("sigma1", np.inf):
                    continue  # resample

            if noise_mode == "three-regime":
                if params.get("sigma0", -np.inf) <= params.get("sigma1", np.inf) and params.get("sigma1", -np.inf) <= params.get("sigma2", np.inf):
                    continue  # resample
    
            break  # valid

        params["noise_mode"] = noise_mode
        params["metric"] = metric
        params["layer"] = layer
        params["encoder"] = encoder_name
        params["stimulus_set"] = hr_task_name

        # -------------------------
        # RUN MODEL
        # -------------------------
        k = min(subsample, len(experiment_list))
        experiment_list_copy = list(experiment_list) 
        
        if random_draw:# avoid mutating original
            rng.shuffle(experiment_list_copy)
        experiment_list_sub = experiment_list_copy[:k]
        
        run_out = run_experiment_scores(
            sigma0=params["sigma0"],
            sigma1=params.get("sigma1", None),
            sigma2=params.get("sigma2", None),
            t_step=params.get("t_step", None),
            rate=params.get("rate", None),
            noise_mode=noise_mode,
            metric=metric,
            X0=X0,
            name_to_idx=name_to_idx,
            experiment_list=experiment_list_sub,
            debug=debug,
            seed=seed*2+1
        )

        model_dp = compute_model_dprime_for_run(run_out, isis)

        
        # ---- determine regime split ----
        if "t_step" in params:
            t_step = params["t_step"]
        else:
            t_step = np.inf
        
        early_end, late_start = isi_split_indices(isis, t_step)
        
        # ---- compute regime-specific errors ----

        mse_zero = compute_mse(
            model_dp, human_curve,
            start_idx=0,
            end_idx=1
        )
        
        mse_early = compute_mse(
            model_dp, human_curve,
            start_idx=0,
            end_idx=early_end
        )

        mse_early_no_zero = compute_mse(
            model_dp, human_curve,
            start_idx=1,
            end_idx=early_end
        )
        
        mse_late = compute_mse(
            model_dp, human_curve,
            start_idx=late_start,
            end_idx=-1
        )
        
        mse_all = compute_mse(model_dp, human_curve)

        try:
            cosine_sim = cosine_similarity_dp(model_dp, human_curve)
            per_isi_mse_vec, per_isi_mse_dict = mse_per_isi(model_dp, human_curve, isis=isis)
        except ValueError:
            cosine_sim = 1
            per_isi_mse_vec, per_isi_mse_dic = [], {}

        
        all_results.append({
            "params": params,
            "results": run_out,
            "model_dp": model_dp,
            "mse_early": mse_early,
            "mse_zero": mse_zero,
            "mse_early_no_zero": mse_early_no_zero,
            "mse_late": mse_late,
            "mse_all": mse_all,
            "cosine_sim": cosine_sim,
            "mse_per_isi": per_isi_mse_vec,          # aligned with `isis`
            "mse_per_isi_dict": per_isi_mse_dict,  
        })
    
    return all_results
        
def random_search_gridlike_v2(
    n_samples,
    param_bounds,
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
    all_results = []

    for i in range(n_samples):
        while True:
            params = {}
            for key, bound in param_bounds.items():
                lo, hi = bound
                params[key] = lo if lo == hi else np.random.uniform(lo, hi)
    
            if noise_mode == "two-regime":
                if params.get("sigma0", -np.inf) <= params.get("sigma1", np.inf):
                    continue  # resample

            if noise_mode == "three-regime":
                if params.get("sigma0", -np.inf) <= params.get("sigma1", np.inf) and params.get("sigma1", -np.inf) <= params.get("sigma2", np.inf):
                    continue  # resample
    
            break  # valid

        params["noise_mode"]   = noise_mode
        params["metric"]       = metric
        params["layer"]        = layer
        params["encoder"]      = encoder_name
        params["stimulus_set"] = hr_task_name

        # -------------------------
        # RUN MODEL
        # -------------------------
        run_out = run_experiment_scores_v2(
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

        nmse = compute_nmse(model_dp, human_curve)
        nmse_no_0 = compute_nmse(model_dp, human_curve, start_idx=1)
        mse = compute_mse(model_dp, human_curve)
        mse_no_0 = compute_mse(model_dp, human_curve, start_idx=1)

        all_results.append({
            "params": params,      
            "results": run_out,
            "model_dp": model_dp,
            "nmse": nmse,
            "nmse_no_0": nmse_no_0,
            "mse": mse,
            "mse_no_0": mse_no_0
        })

    return all_results

def best_model_by_layer_metric(grid_results, best_key="nmse"):
    best = {}

    for rec in grid_results:
        p = rec["params"]
        metric = p["metric"] 
        layer = p["layer"]
        key = (metric, layer)

        current_val = rec[best_key]
        if key not in best or current_val < best[key][best_key]:
            best[key] = rec

    return best

########################
# E. PLOTTING
########################

def generate_and_plot_model_decay_summary_v2(
    grid_results,
    human_curve,
    isis,
    savedir=None,
    max_rows=6,
    verbose=True,
    hr_task_name="", encoder_name=""
):
    """
    New version of the decay summary plotter.
    Compatible with runners_v2 results.
    """

    # -----------------------------
    # 1. GROUP MODELS (metric × noise_mode)
    # -----------------------------
    grouped = {}   # (metric, noise_mode) → list of (params, results)

    for rec in grid_results:
        p = rec["params"]
        metric = p.get("metric", "unknown")
        noise_mode = p.get("noise_mode", "unknown")
        layer = p.get("layer", "unknown")
        
        if noise_mode == "two-regime" or noise_mode == "three-regime":
            t_step = p.get("t_step", None)
            key = (metric, noise_mode, layer, t_step)
        else:
            key = (metric, noise_mode, layer)

        grouped.setdefault(key, []).append(rec)

    keys = sorted(grouped.keys())

    # Determine how many panels
    n_models = len(keys)
    ncols = 2
    nrows = int(np.ceil(n_models / ncols))
    nrows = min(nrows, max_rows)

    # -----------------------------
    # 2. FIRST FIGURE: ALL d' CURVES
    # -----------------------------
    fig1, axes1 = plt.subplots(nrows, ncols, figsize=(6*ncols, 3.5*nrows), squeeze=False)

    best_fits = {}
    human_dp = human_curve.astype(float)

    flat_ax = axes1.flatten()

    for idx, key in enumerate(keys):
        
        if len(key) == 4:
            metric, noise_mode, layer, t_step = key
            title_tag = f"layer={layer} | t={t_step}"
        else:
            metric, noise_mode, layer = key
            t_step = None
            title_tag = f"layer={layer}"
            
        ax = flat_ax[idx]

        models = grouped[key]

        # For x-axis grouping: sort by sigma0 or rate depending on noise_mode
        def sort_key(r):
            p = r["params"]
            # prioritize sigma0 first
            return (p.get("sigma0", np.nan), p.get("rate", np.nan), p.get("sigma1", np.nan))

        models_sorted = sorted(models, key=sort_key)

        nmse_list = []
        dp_list = []
        param_list = []

        for rec in models_sorted:
            params = rec["params"]
            run_out = rec["results"]

            model_dp = compute_model_dprime_for_run(run_out, isis)
            dp_list.append(model_dp)
            param_list.append(params)

            # Compute NMSE
            nmse = compute_nmse(model_dp, human_dp)
            nmse_list.append(nmse)

            lbl = ", ".join(f"{k}={v}" for k,v in params.items())
            ax.plot(isis, model_dp, "-o", alpha=0.5, label=lbl)

        ax.plot(isis, human_dp, "o-k", label="Human", linewidth=3)
        ax.set_title(f"{metric} | {noise_mode} | {title_tag}")
        ax.set_xlabel("ISI")
        ax.set_ylabel("d′")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=2)

        # Store best fit index
        best_idx = int(np.nanargmin(nmse_list))
        best_fits[key] = {
            "params": param_list[best_idx],
            "model_dp": dp_list[best_idx],
            "nmse": float(nmse_list[best_idx]),
            "run_out": models_sorted[best_idx]["results"],   # added line
        }

        best_fits[key]["layer"] = param_list[best_idx].get("layer")
        best_fits[key]["encoder"] = param_list[best_idx].get("encoder")
        best_fits[key]["stimulus_set"] = param_list[best_idx].get("stimulus_set")
        best_fits[key]["t_step"] = t_step

    plt.suptitle(f"{hr_task_name}: All Model d′ vs ISI Curves - Encoder: {encoder_name}", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.97])
    if savedir:
        plt.savefig(savedir + "/model_decay_all_curves.png", dpi=200)
    plt.show()


    # -----------------------------
    # 4. THIRD FIGURE: Best-Fit Model vs Human
    # -----------------------------
    fig3, axes3 = plt.subplots(nrows, ncols, figsize=(6*ncols, 3.5*nrows), squeeze=False)
    flat_ax3 = axes3.flatten()

    for idx, key in enumerate(keys):
        
        if len(key) == 4:
            metric, noise_mode, layer, t_step = key
            title_tag = f"layer={layer} | t={t_step}"
        else:
            metric, noise_mode, layer = key
            t_step = None
            title_tag = f"layer={layer}"
        ax = flat_ax3[idx]

        if key not in best_fits:
            ax.axis("off")
            continue

        best = best_fits[key]
        params = best["params"]
        model_dp = best["model_dp"]
        nmse = best["nmse"]

        ax.plot(isis, human_dp, "o-k", label="Human", linewidth=3)
        ax.plot(isis, model_dp, "s--", color="tab:blue", label="Best Model")

        txt = ", ".join(f"{k}={v:.2f}" for k,v in params.items()  if "sigma" in k)
        ax.set_title(
            f"{metric} | {noise_mode}\n{title_tag}\n{txt}\nNMSE={nmse:.3f}"
        )
        ax.set_xlabel("ISI")
        ax.set_ylabel("d′")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plt.suptitle(f"{hr_task_name}: Best-Fit Models vs Human - Encoder: {encoder_name}", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.97])
    if savedir:
        plt.savefig(savedir + "/model_decay_bestfits.png", dpi=200)
    plt.show()

    return best_fits


def generate_and_plot_model_decay_summary_v5(
    grid_results,
    human_curve,
    isis,
    *,
    metric_name="mse_per_isi",
    isi_indices=None,
    savedir=None,
    max_rows=6,
    hr_task_name="",
    encoder_name="",
):
    """
    Model decay summary with breathing room:

    For each model family:
      Row A: large d′ vs ISI (all models + best + human)
      Row B: metric vs sigma0 / sigma1 / sigma2
    """

    # -----------------------------
    # Helper: sigma diagnostic plot
    # -----------------------------
    def plot_sigma_metric(ax, summaries, sigma_key, metric_name, best_sigma):
        xs, ys = [], []

        for s in summaries:
            if sigma_key in s["params"]:
                xs.append(s["params"][sigma_key])
                ys.append(s["score"])

        if len(xs) == 0:
            ax.axis("off")
            return

        xs = np.asarray(xs)
        ys = np.asarray(ys)

        # jitter for overplotting
        jitter = 0.01 * np.random.randn(len(xs))
        ax.scatter(xs + jitter, ys, alpha=0.6)

        ax.axvline(best_sigma, color="tab:red", linestyle="--", alpha=0.8)

        ax.set_xlabel(sigma_key)
        ax.set_ylabel(metric_name)
        ax.set_title(sigma_key)
        ax.grid(True, alpha=0.3)

    # -----------------------------
    # Group models
    # -----------------------------
    grouped = {}
    for rec in grid_results:
        p = rec["params"]
        metric = p.get("metric", "unknown")
        noise_mode = p.get("noise_mode", "unknown")
        layer = p.get("layer", "unknown")

        if noise_mode in {"two-regime", "three-regime"}:
            key = (metric, noise_mode, layer, p.get("t_step"))
        else:
            key = (metric, noise_mode, layer)

        grouped.setdefault(key, []).append(rec)

    keys = sorted(grouped.keys())[:max_rows]

    human_dp = human_curve.astype(float)

    # -----------------------------
    # Figure + GridSpec
    # -----------------------------
    n_models = len(keys)
    fig = plt.figure(figsize=(15, 6.5 * n_models))

    gs = GridSpec(
        nrows=2 * n_models,
        ncols=3,
        figure=fig,
        height_ratios=[2.4, 1.6] * n_models,
        hspace=0.5,
        wspace=0.35,
    )

    best_fits = {}

    # -----------------------------
    # Main loop
    # -----------------------------
    for i, key in enumerate(keys):

        row_curve = 2 * i
        row_sigma = 2 * i + 1

        ax_curve = fig.add_subplot(gs[row_curve, :])
        ax_s0 = fig.add_subplot(gs[row_sigma, 0])
        ax_s1 = fig.add_subplot(gs[row_sigma, 1])
        ax_s2 = fig.add_subplot(gs[row_sigma, 2])

        if len(key) == 4:
            metric, noise_mode, layer, t_step = key
            title_tag = f"layer={layer} | t={t_step}"
        else:
            metric, noise_mode, layer = key
            t_step = None
            title_tag = f"layer={layer}"

        models = grouped[key]
        summaries = []
        scores = []

        # ---- collect metrics ----
        for rec in models:
            params = rec["params"]
            run_out = rec["results"]

            model_dp = compute_model_dprime_for_run(run_out, isis)
            mse_per_isi = (model_dp - human_dp) ** 2
            nmse = compute_nmse(model_dp, human_dp)

            if isi_indices is None:
                scalar_mse = float(np.mean(mse_per_isi))
            else:
                scalar_mse = float(np.mean(mse_per_isi[isi_indices]))

            score = scalar_mse if metric_name == "mse_per_isi" else nmse

            summaries.append({
                "params": params,
                "model_dp": model_dp,
                "mse_per_isi": mse_per_isi,
                "mse_early": float(np.mean(mse_per_isi[:2])),
                "mse_late": float(np.mean(mse_per_isi[2:])),
                "nmse": float(nmse),
                "score": float(score),
                "run_out": run_out,
            })

            scores.append(score)

            # background curves
            ax_curve.plot(
                isis,
                model_dp,
                "-",
                color="gray",
                alpha=0.25,
                linewidth=1,
            )

        # ---- best model ----
        best_idx = int(np.nanargmin(scores))
        best = summaries[best_idx]
        best_fits[key] = best

        ax_curve.plot(
            isis,
            human_dp,
            "o-k",
            linewidth=3,
            label="Human",
        )
        ax_curve.plot(
            isis,
            best["model_dp"],
            "s--",
            color="tab:blue",
            linewidth=2.5,
            label="Best model",
        )

        sigma_txt = ", ".join(
            f"{k}={v:.2f}"
            for k, v in best["params"].items()
            if "sigma" in k
        )

        ax_curve.set_title(
            f"{metric} | {noise_mode} | {title_tag}\n"
            f"{sigma_txt}   {metric_name}={best['score']:.3f}",
            fontsize=11,
        )
        ax_curve.set_xlabel("ISI")
        ax_curve.set_ylabel("d′")
        ax_curve.set_ylim(bottom=0)
        ax_curve.grid(True, alpha=0.3)
        ax_curve.legend(fontsize=8)

        # ---- sigma diagnostics (row 2) ----
        plot_sigma_metric(ax_s0, summaries, "sigma0", metric_name, best["params"].get("sigma0"))
        plot_sigma_metric(ax_s1, summaries, "sigma1", metric_name, best["params"].get("sigma1"))
        plot_sigma_metric(ax_s2, summaries, "sigma2", metric_name, best["params"].get("sigma2"))

    # -----------------------------
    # Global title + save
    # -----------------------------
    fig.suptitle(
        f"{hr_task_name}: Model Decay Summary ({metric_name})\n\n\n",
        fontsize=16,
    )

    if savedir:
        plt.savefig(f"{savedir}/{encoder_name}-model_decay_summary.png", dpi=200, bbox_inches="tight")

    plt.show()

    return best_fits


def generate_and_plot_compact_summary(
    final_result,
    human_curve,
    isis,
    *,
    t_step=5,
    savedir=None,
    hr_task_name="",
    encoder_name="",
):
    """
    Compact 2-panel summary for a single fitted run.

    Left panel:
        d' vs ISI (model vs human) with regime shading.
    Right panel:
        Per-ISI MSE bars with overall MSE reference line.

    Returns
    -------
    dict
        ``best_fits``-style dictionary compatible with ``save_best_models``.
    """
    human_dp = np.asarray(human_curve, dtype=float)
    model_dp = compute_model_dprime_for_run(final_result["results"], isis)
    mse_per_isi = (model_dp - human_dp) ** 2

    params = dict(final_result.get("params", {}))
    sigma0 = params.get("sigma0", np.nan)
    sigma1 = params.get("sigma1", np.nan)
    sigma2 = params.get("sigma2", np.nan)

    mse_all = float(np.mean(mse_per_isi))
    early_mask = np.asarray([(1 <= isi <= 4) for isi in isis], dtype=bool)
    late_mask = np.asarray([(isi >= 8) for isi in isis], dtype=bool)

    mse_isi0 = float(mse_per_isi[isis.index(0)]) if 0 in isis else np.nan
    mse_early = float(np.mean(mse_per_isi[early_mask])) if np.any(early_mask) else np.nan
    mse_late = float(np.mean(mse_per_isi[late_mask])) if np.any(late_mask) else np.nan
    nmse = float(compute_nmse(model_dp, human_dp))

    rows = [
        ("sigma0", sigma0),
        ("sigma1", sigma1),
        ("sigma2", sigma2),
        ("MSE(ISI=0)", mse_isi0),
        ("MSE(ISI 1-4)", mse_early),
        ("MSE(ISI 8-64)", mse_late),
        ("MSE(all)", mse_all),
    ]

    print("\n" + "=" * 52)
    print("Compact three-stage fit summary")
    print("=" * 52)
    for k, v in rows:
        print(f"{k:14s}: {v:.6f}" if np.isfinite(v) else f"{k:14s}: nan")
    print("=" * 52 + "\n")

    x = np.arange(len(isis))
    fig, (ax_curve, ax_mse) = plt.subplots(1, 2, figsize=(12.5, 4.2))

    def _regime_color(isi):
        if isi == 0:
            return "tab:green"
        if isi < t_step:
            return "tab:orange"
        return "tab:purple"

    for i, isi in enumerate(isis):
        if isi == 0:
            ax_curve.axvspan(i - 0.5, i + 0.5, color="tab:green", alpha=0.08)
        elif isi < t_step:
            ax_curve.axvspan(i - 0.5, i + 0.5, color="tab:orange", alpha=0.08)
        else:
            ax_curve.axvspan(i - 0.5, i + 0.5, color="tab:purple", alpha=0.08)

    ax_curve.plot(x, human_dp, "o-k", linewidth=2.3, label="Human")
    ax_curve.plot(x, model_dp, "s-", color="tab:blue", linewidth=2.0, label="Model")
    ax_curve.set_xticks(x)
    ax_curve.set_xticklabels([str(v) for v in isis])
    ax_curve.set_xlabel("ISI")
    ax_curve.set_ylabel("d′")
    ax_curve.set_title("d′ vs ISI")
    ax_curve.grid(True, alpha=0.25)
    ax_curve.legend(frameon=False)

    ymax = max(np.nanmax(human_dp), np.nanmax(model_dp))
    ytxt = ymax + 0.05 * max(1.0, ymax)
    if 0 in isis:
        i0 = isis.index(0)
        ax_curve.text(i0, ytxt, r"$\sigma_0$", ha="center", va="bottom", fontsize=9)
    sigma1_idx = [i for i, isi in enumerate(isis) if 1 <= isi < t_step]
    if sigma1_idx:
        ax_curve.text(np.mean(sigma1_idx), ytxt, r"$\sigma_1$", ha="center", va="bottom", fontsize=9)
    sigma2_idx = [i for i, isi in enumerate(isis) if isi >= t_step]
    if sigma2_idx:
        ax_curve.text(np.mean(sigma2_idx), ytxt, r"$\sigma_2$", ha="center", va="bottom", fontsize=9)

    bar_colors = [_regime_color(isi) for isi in isis]
    ax_mse.bar(x, mse_per_isi, color=bar_colors, edgecolor="black", linewidth=0.3)
    ax_mse.axhline(mse_all, color="black", linestyle="--", linewidth=1.2, label=f"Overall={mse_all:.3f}")
    ax_mse.set_xticks(x)
    ax_mse.set_xticklabels([str(v) for v in isis])
    ax_mse.set_xlabel("ISI")
    ax_mse.set_ylabel("MSE")
    ax_mse.set_title("Per-ISI MSE")
    ax_mse.grid(axis="y", alpha=0.25)
    ax_mse.legend(frameon=False, fontsize=8)

    title = f"{hr_task_name}: compact three-stage fit\n{encoder_name}"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])

    if savedir:
        Path(savedir).mkdir(parents=True, exist_ok=True)
        png_path = Path(savedir) / f"{encoder_name}-compact_summary.png"
        txt_path = Path(savedir) / f"{encoder_name}-compact_summary.txt"
        fig.savefig(png_path, dpi=200, bbox_inches="tight")
        with open(txt_path, "w") as f:
            f.write("Compact three-stage fit summary\n")
            f.write("=" * 52 + "\n")
            for k, v in rows:
                f.write(f"{k:14s}: {v:.6f}\n" if np.isfinite(v) else f"{k:14s}: nan\n")
            f.write("=" * 52 + "\n")

    plt.show()

    metric = params.get("metric", "mse_per_isi")
    noise_mode = params.get("noise_mode", "three-regime")
    layer = params.get("layer", encoder_name)
    key = (metric, noise_mode, layer, params.get("t_step", t_step))

    best_fits = {
        key: {
            "params": params,
            "model_dp": np.asarray(model_dp),
            "mse_per_isi": np.asarray(mse_per_isi),
            "mse_early": mse_early,
            "mse_late": mse_late,
            "nmse": nmse,
            "score": mse_all,
            "run_out": final_result["results"],
        }
    }

    return best_fits

def plot_noise_schedules(grid_results, max_plots=8):
    """
    Plot std(age) curves for multiple runs.
    """
    plt.figure(figsize=(8,6))

    for i, rec in enumerate(grid_results[:max_plots]):
        params = rec["params"]
        stds = rec["results"]["stds_over_time"]

        if len(stds) == 0:
            continue
        
        t = stds[:,0]
        s = stds[:,1]

        label = ", ".join(f"{k}={v:.2f}" for k,v in params.items())
        plt.scatter(t, s, alpha=0.3, label=label)

    plt.xlabel("Time (t)")
    plt.ylabel("Noise Std (σ)")
    plt.title("Noise Schedule Over Time")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_hit_fa_distributions(grid_results, max_plots=4, bins=40, hr_task_name="", encoder_name=""):
    import matplotlib.pyplot as plt

    n = min(len(grid_results), max_plots)
    cols = 2
    rows = n

    fig, axes = plt.subplots(rows, cols, figsize=(11, 3.6 * rows))

    if rows == 1:
        axes = np.array([axes])

    row_labels = []
    for i in range(n):
        params = grid_results[i]["params"]
        row_labels.append(
            f"$\sigma_0$={params['sigma0']:.2f}  |  $\sigma_1$={params['sigma1']:.2f}  |  mode={params['noise_mode']}"
        )

    # -------------------------------------------------------
    # 1. PLOT HISTOGRAMS
    # -------------------------------------------------------
    for i in range(n):
        rec = grid_results[i]
        res = rec["results"]

        hits = res.get("hits", [])
        fas  = res.get("fas", [])

        ax_h = axes[i, 0]
        ax_f = axes[i, 1]

        # hits
        ax_h.hist(hits, bins=bins, color="royalblue", alpha=0.75)
        ax_h.set_title("Hits")
        ax_h.set_xlabel("Score")
        ax_h.set_ylabel("Count")

        # fas
        ax_f.hist(fas, bins=bins, color="tomato", alpha=0.75)
        ax_f.set_title("False Alarms")
        ax_f.set_xlabel("Score")
        ax_f.set_ylabel("Count")

    # -------------------------------------------------------
    # 2. RUN LAYOUT FIRST
    # -------------------------------------------------------
    fig.tight_layout()

    # Important: draw so positions update
    fig.canvas.draw()

    # -------------------------------------------------------
    # 3. Now compute & place row labels ABOVE each row
    # -------------------------------------------------------
    for i in range(rows):
        ax_left  = axes[i, 0]
        ax_right = axes[i, 1]

        # get final positions
        left_pos  = ax_left.get_position()
        right_pos = ax_right.get_position()

        # center x: midpoint between the two axes
        x_center = (left_pos.x0 + right_pos.x1) / 2

        # y position: slightly above the tallest of the two axes
        y_top = max(left_pos.y1, right_pos.y1) + 0.01

        fig.text(
            x_center,
            y_top,
            row_labels[i],
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="medium"
        )

    # -------------------------------------------------------
    # 4. Add global title
    # -------------------------------------------------------
    fig.suptitle(f"{hr_task_name}: Model Hit/FA Score Distributions - Encoder: {encoder_name}", fontsize=18, y=1.02)

    plt.show()


def plot_noise_schedule_function(noise_mode, params, max_age=50):
    """
    Plot std(age) for a given noise schedule, ignoring the simulation.
    """
    schedule = make_noise_schedule(noise_mode, params)
    ages = np.arange(1, max_age + 1)
    stds = [schedule(a) for a in ages]

    plt.figure(figsize=(6,4))
    plt.plot(ages, stds, marker="o")
    plt.xlabel("Age")
    plt.ylabel("Noise std (σ)")
    title = f"Noise schedule: {noise_mode}, " + ", ".join(f"{k}={v}" for k,v in params.items())
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_model_vs_human_curves(grid_results, human_curve, isis, max_plots=6, hr_task_name="", encoder_name=""):
    """
    Plot d′ curve for human vs a few model runs.
    """
    plt.figure(figsize=(8,6))
    plt.plot(isis, human_curve, "o-k", label="Human", linewidth=3)

    for i, rec in enumerate(grid_results[:max_plots]):
        params = rec["params"]
        label = ", ".join(f"{k}={v}" for k,v in params.items())
        
        model_dp = compute_model_dprime_for_run(rec["results"], isis)
        
        plt.plot(isis, model_dp, "-o", alpha=0.6, label=label)

    plt.xlabel("ISI (s)")
    plt.ylabel("d′")
    plt.grid(True, alpha=0.3)
    plt.title(f"{hr_task_name}: Model vs Human d′ Curves\n Encoder: {encoder_name}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_best_models(best_models, human_curve, isis, hr_task_name="", encoder_name=""):
    plt.figure(figsize=(8,6))
    plt.plot(isis, human_curve, "o-k", linewidth=3, label="Human")

    for (metric, noise_mode), rec in best_models.items():
        label = f"{metric} | {noise_mode}  (NMSE={rec['nmse']:.3f})"
        plt.plot(isis, rec["model_dp"], "-o", alpha=0.7, label=label)

    plt.xlabel("ISI (s)")
    plt.ylabel("d′")
    plt.grid(True, alpha=0.3)
    plt.title(f"{hr_task_name}: Best Model per Class (metric × noise_mode)\n Encoder: {encoder_name}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

def plot_best_model_histograms(best_fits, isis, savedir=None, bins=40):
    """
    Plot histograms of hit scores per ISI and temporal false alarms
    for each best model (metric × noise_mode × layer × [t_step]).
    """

    for key, rec in best_fits.items():

        # ----------------------------
        # Unpack model identity safely
        # ----------------------------
        if len(key) == 4:
            metric, noise_mode, layer, t_step = key
            title_tag = f"layer={layer} | t={t_step}"
            noise_tag = f"{noise_mode}_t{t_step}"
        else:
            metric, noise_mode, layer = key
            t_step = None
            title_tag = f"layer={layer}"
            noise_tag = noise_mode

        params   = rec["params"]
        nmse     = rec["nmse"]
        run_out  = rec["run_out"]

        isi_hit_dists = run_out["isi_hit_dists"]
        fa_by_t       = run_out["fa_by_t"]
        T_max         = run_out["T_max"]

        # ============================================================
        # 1. HIT HISTOGRAMS PER ISI
        # ============================================================
        n = len(isis)
        ncols = 4
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(4*ncols, 3*nrows),
            squeeze=False
        )
        axs = axes.flatten()

        for i, isi in enumerate(isis):
            ax = axs[i]

            entries = isi_hit_dists.get(isi+1, [])
            hit_scores = np.array([score for (score, _) in entries], float)

            if len(hit_scores) > 0:
                ax.hist(hit_scores, bins=bins, alpha=0.85, color='g')
                mu = np.mean(hit_scores)
                ax.set_title(rf"ISI={isi}: $\mu={mu:.3f}$", fontsize=10)
            else:
                ax.text(0.5, 0.5, "No hits", ha="center", va="center")
                ax.set_title(f"ISI={isi}")

            ax.set_xlabel("Score")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"Hit Score Distributions\n"
            f"(metric={metric}, noise={noise_tag}, {title_tag})\n"
            f"NMSE={nmse:.3f}",
            fontsize=14
        )
        fig.tight_layout(rect=[0, 0, 1, 0.92])

        if savedir:
            plt.savefig(
                f"{savedir}/bestmodel_hist_hits_{metric}_{noise_tag}_{layer}.png",
                dpi=200
            )

        plt.show()

        # ============================================================
        # 2. FALSE ALARM HISTOGRAMS BY EXPERIMENT POSITION
        # ============================================================
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        titles = ["Early FAs", "Middle FAs", "Late FAs"]

        fa_scores_early, fa_scores_mid, fa_scores_late = [], [], []

        midpoint = T_max // 2
        window = 10

        for t in range(1, T_max + 1):
            scores_t = fa_by_t[t-1]
            if t <= window:
                fa_scores_early.extend(scores_t)
            elif abs(t - midpoint) <= window:
                fa_scores_mid.extend(scores_t)
            elif t >= T_max - window:
                fa_scores_late.extend(scores_t)

        buckets = [fa_scores_early, fa_scores_mid, fa_scores_late]

        for ax, scores, title in zip(axes, buckets, titles):
            scores = np.asarray(scores, float)
            if len(scores) > 0:
                ax.hist(scores, bins=bins, alpha=0.85, color='r')
                mu = np.mean(scores)
                ax.set_xlabel("FA Score")
                ax.set_title(f"{title} ($\mu={mu:.3f}$)")
            else:
                ax.text(0.5, 0.5, "No FAs", ha="center", va="center")
                ax.set_title(title)

            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)

        fig.suptitle(
            f"False Alarm Score Evolution\n"
            f"(metric={metric}, noise={noise_tag}, {title_tag})\n"
            f"NMSE={nmse:.3f}",
            fontsize=14
        )
        fig.tight_layout(rect=[0, 0, 1, 0.92])

        if savedir:
            plt.savefig(
                f"{savedir}/bestmodel_hist_fas_{metric}_{noise_tag}_{layer}.png",
                dpi=200
            )

        plt.show()

## saving

import json, os, numpy as np
import pickle


def save_best_models(best_fits, save_dir, prefix="bestmodels"):
    """
    Save best model summaries and full objects.

    - Pickle: authoritative full record
    - JSON: human-readable summary with all key metrics
    - NPZ: numeric convenience (model_dp + scalar metrics)
    """

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    summary_list = []

    for key, rec in best_fits.items():

        # ----------------------------
        # Unpack model identity
        # ----------------------------
        if len(key) == 4:
            metric, noise_mode, layer, t_step = key
        else:
            metric, noise_mode, layer = key
            t_step = None

        params = rec["params"]
        model_dp = rec["model_dp"]
        run_out = rec["run_out"]

        # metrics (robust access)
        score = rec.get("score")
        mse_early = rec.get("mse_early")
        mse_late = rec.get("mse_late")
        mse_per_isi = rec.get("mse_per_isi")
        nmse = rec.get("nmse", None)  # legacy, optional

        encoder = params.get("encoder", "unknown")
        stimset = params.get("stimulus_set", "unknown")

        # ----------------------------
        # JSON summary (explicit & honest)
        # ----------------------------
        summary_list.append({
            "metric_family": metric,
            "noise_mode": noise_mode,
            "t_step": t_step,
            "layer": layer,
            "encoder": encoder,
            "stimulus_set": stimset,

            # optimization target
            "score": score,

            # diagnostics
            "mse_early": mse_early,
            "mse_late": mse_late,

            # light payloads
            "params": params,
            "model_dp": model_dp.tolist(),
        })

        # ----------------------------
        # Filename-safe noise tag
        # ----------------------------
        if noise_mode in {"two-regime", "three-regime"}:
            assert t_step is not None, f"{noise_mode} model missing t_step"
            noise_tag = f"{noise_mode}_t{t_step}"
        else:
            noise_tag = noise_mode

        tag = f"{prefix}_{metric}_{noise_tag}"

        # ----------------------------
        # Pickle (authoritative)
        # ----------------------------
        with open(save_dir / f"{tag}.pkl", "wb") as f:
            pickle.dump(rec, f)

        # ----------------------------
        # NPZ (numeric convenience)
        # ----------------------------
        np.savez_compressed(
            save_dir / f"{tag}.npz",
            model_dp=model_dp,
            score=score,
            mse_early=mse_early,
            mse_late=mse_late,
            mse_per_isi=mse_per_isi,
            nmse=nmse,
        )

    # ----------------------------
    # Master JSON
    # ----------------------------
    with open(save_dir / f"{prefix}_summary.json", "w") as f:
        json.dump(summary_list, f, indent=2)

    print(f"✔ Saved {len(best_fits)} best models to {save_dir}")
    return summary_list
