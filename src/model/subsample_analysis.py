# ===================== Imports =====================
import argparse, sys, os, glob, json, math, datetime, torch, yaml
import matplotlib.pyplot as plt, numpy as np, pandas as pd

from collections import defaultdict
from scipy.stats import norm, pearsonr

from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from pathlib import Path
from matplotlib.gridspec import GridSpec

from scipy.spatial.distance import pdist

from tqdm import tqdm_notebook
from tqdm.notebook import trange, tqdm

sys.path.append('/om2/user/jmhicks/projects/TextureStreaming/code/')
sys.path.append('/om2/user/bjmedina/auditory-memory/memory/utls/')
sys.path.append('/om2/user/bjmedina/auditory-memory/memory/src/model/')
sys.path.append("/om2/user/bjmedina/auditory-memory/memory/")

from chexture_choolbox.auditorytexture.texture_model import TextureModel
from chexture_choolbox.auditorytexture.helpers import FlattenStats
from texture_prior.params import model_params, statistics_set, texture_dataset
from texture_prior.utils import path

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
from encoders import *


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
        return params["sigma0"] > params["sigma1"]
    if noise_mode == "three-regime":
        # enforce sigma0 > sigma1 > sigma2
        return params["sigma0"] > params["sigma1"] > params["sigma2"]
    return True

def make_grid(lo, hi, n, *, spacing="log"):
    """
    Create 1D grid between lo and hi.

    spacing:
        - "log"    : log-spaced
        - "linear" : linearly spaced
        - "hybrid" : half log, half linear
    """
    lo = float(lo)
    hi = float(hi)

    if n <= 1 or lo == hi:
        return [lo]

    if spacing == "log":
        lo = max(lo, 1e-12)
        hi = max(hi, lo * 1.0001)
        return list(np.exp(np.linspace(np.log(lo), np.log(hi), n)))

    if spacing == "linear":
        return list(np.linspace(lo, hi, n))

    if spacing == "hybrid":
        n1 = n // 2
        n2 = n - n1
        log_part = np.exp(
            np.linspace(np.log(max(lo, 1e-12)), np.log(hi), n1, endpoint=False)
        )
        lin_part = np.linspace(lo, hi, n2)
        return sorted(set(log_part.tolist() + lin_part.tolist()))

    raise ValueError(f"Unknown spacing: {spacing}")

def generate_candidates(
    param_bounds,
    *,
    free_keys,
    fixed_params=None,
    n_per_dim=5,
    noise_mode="three-regime",
    spacing="log",          # NEW
    spacing_by_key=None,    # OPTIONAL PER-PARAM CONTROL
):
    """
    Build deterministic param dicts.

    spacing:
        "log", "linear", or "hybrid"

    spacing_by_key:
        dict like {"sigma0": "linear", "sigma1": "log"}
    """
    fixed_params = {} if fixed_params is None else dict(fixed_params)
    spacing_by_key = {} if spacing_by_key is None else dict(spacing_by_key)

    values_by_key = {}
    for k, (lo, hi) in param_bounds.items():
        if k in fixed_params:
            values_by_key[k] = [float(fixed_params[k])]
        elif k in free_keys:
            sp = spacing_by_key.get(k, spacing)
            values_by_key[k] = make_grid(lo, hi, n_per_dim, spacing=sp)
        else:
            if lo != hi:
                raise ValueError(
                    f"Key '{k}' not in free_keys and not fixed_params, but bounds vary: {(lo, hi)}"
                )
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

model_cfg, model_cfg_path = load_config("/om2/user/bjmedina/auditory-memory/memory/model_yamls/three-regime/resnet50/nontime_avg/run_000005.yaml")

noise_cfg = model_cfg["noise_model"]
#print(model_cfg['experiment']['n_seqs'])
print(model_cfg, model_cfg_path)
print(model_cfg['representation']['type'], model_cfg['representation']['layer'])

assert "t_step" in noise_cfg, "t_step is needed for two-regime"

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
print(noise_cfg)

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
    which_task, which_isi, is_multi, old=False)

# ---------------------------
# 2. Human curve
# ---------------------------
human_curve = compute_human_curve(human_runs, is_multi, which_isi)
results_root = model_cfg["results_root"]
tag = model_cfg.get("tag", "untagged")

if noise_mode == "two-regime" or noise_mode == "three-regime":
    assert "t_step" in noise_cfg, f"{noise_mode} requires t_step"
    t_step = noise_cfg["t_step"]
    noise_tag = f"{noise_mode}_t{t_step}"
else:
    noise_tag = noise_mode

import time


if time_avg:
    save_figs = (
        f"{results_root}/"
        f"figures/v13_three-regime_time_avg/"
        f"{task_name}/"
        f"{encoder_type}/"
        f"{metric}/"
        f"{noise_tag}/"
    )
    
    save_fits = f"{results_root}/fits/v13_three-regime_time_avg"
    save_fits_all = f"{results_root}/fits/v13_three-regime_time_avg_all"
    
    ensure_dir(save_figs)
    ensure_dir(save_fits_all)
    ensure_dir(save_fits)
else:
    save_figs = (
        f"{results_root}/"
        f"figures/v13_three-regime_nontime_avg/"
        f"{task_name}/"
        f"{encoder_type}/"
        f"{metric}/"
        f"{noise_tag}/"
    )
    
    save_fits = f"{results_root}/fits/v13_three-regime_nontime_avg"
    save_fits_all = f"{results_root}/fits/v13_three-regime_nontime_avg_all"
    
    ensure_dir(save_figs)
    ensure_dir(save_fits_all)
    ensure_dir(save_fits)  

encoder_type = repr_cfg["type"]     # e.g. "resnet50"
layer        = repr_cfg.get("layer")
pc_dims      = repr_cfg.get("pc_dims")

print(layer)

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
    pc_dims=pc_dims,
    sr=20000,
    duration=2.0,
    rms_level=0.05,
    time_avg=time_avg,
    device="cuda",
)

# ---- representation-specific fields ----
if encoder_type in ("kell2018", "resnet50"):
    encoder_cfg["layer"] = layer
    assert layer is not None, f"layer required for {encoder_type}"

if encoder_type == "texture":
    encoder_cfg["pc_dims"] = pc_dims
    assert pc_dims is not None, "pc_dims required for texture encoder"

encoder_name = make_encoder_name(encoder_cfg)
print("Encoder name:", encoder_name)

encoder = build_encoder(encoder_cfg)
X = encode_stimuli(encoder, all_files)

# X is torch tensor at this point
X_np = X.detach().cpu().numpy()
print(X_np.shape)

d50 = median_pairwise_distance(
    X_np,
    metric="cosine",      # e.g. "cosine", "euclidean", "mahalanobis"
    n_samples=500,
    seed=0,
)

noise_cfg = model_cfg["noise_model"]
noise_mode = noise_cfg["name"]

param_bounds = {
    "sigma0": (noise_cfg["sigma0_min"]*1, noise_cfg["sigma0_max"]*1),
}

if noise_mode == "two-regime":
    param_bounds["sigma1"] = (
        noise_cfg["sigma1_min"]*d50,
        noise_cfg["sigma1_max"]*d50,
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
sigma0_min = 0.1
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
    n_per_dim=20,
    spacing="linear",
)

print(f"there are {len(cands)} candidates") 


from collections import defaultdict
import numpy as np
from tqdm import trange

R =  10          # number of repeated runs per subsample
ISI0_IDX = 0
tol0_band = 0.2

sigma0_by_n = defaultdict(list)
mse_by_n = defaultdict(list)
best_models = defaultdict(list)

num_experiments = int(sys.argv[1])


for rep in range(R):

    print(f"rep {rep}, {num_experiments} experiments")
    resA = []

    rng = np.random.default_rng(rep)

    for c_i in trange(len(cands)):

        c = cands[c_i]

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
            subsample=num_experiments,
            random_draw=True,
            rng=rng# IMPORTANT: allow randomness across reps
        )[0]

        resA.append(resA_single)

    bestA = min(
        resA,
        key=lambda r: float(r["mse_per_isi"][ISI0_IDX])
    )

    best_models[num_experiments].append(bestA)
    sigma0_by_n[num_experiments].append(bestA["params"]["sigma0"])
    mse_by_n[num_experiments].append(float(bestA["mse_per_isi"][ISI0_IDX]))
    
# ---------------------------
# Imports
# ---------------------------
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime


# ---------------------------
# Timestamped filenames
# ---------------------------
timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
if time_avg:
    base = f"{save_fits_all}/sigma0/stability_{timestamp}-{model_cfg['representation']['type']}-{model_cfg['representation']['layer']}-timeavg-{num_experiments}_random_experiments"
else:
    base = f"{save_fits_all}/sigma0/stability_{timestamp}-{model_cfg['representation']['type']}-{model_cfg['representation']['layer']}-nontimeavg-{num_experiments}_random_experiments"
fig_fname = f"{base}.png"

ensure_dir(f"{save_fits_all}/sigma0/")

# ---------------------------
# Load results
# ---------------------------
pkl_fname = f"{base}.pkl"

try:
    with open(pkl_fname, "rb") as f:
        out = pickle.load(f)
except FileNotFoundError:
    pass

# Expecting this structure

# ---------------------------
# Plot
# ---------------------------
plt.figure(figsize=(6, 4))

for n, sigs in sigma0_by_n.items():
    sigs = np.asarray(sigs)
    x = np.full(len(sigs), n)
    #jitter = np.random.uniform(-0.005, 0.005, size=len(sigs))
    plt.scatter(x, sigs, alpha=0.3)
    plt.scatter(n, np.median(sigs), color='r', alpha=.5)

# Median line
xs = sorted(sigma0_by_n.keys())
meds = [np.median(sigma0_by_n[n]) for n in xs]
plt.plot(xs, meds, color="black", linewidth=2, label="median σ₀")

plt.xlabel("Subsample size")
plt.ylabel("Best-fit σ₀ (ISI=0)")
plt.title("Stability of σ₀ vs data size")
plt.legend()
plt.tight_layout()

plt.savefig(fig_fname, dpi=200)
plt.show()

print(f"Saved figure to: {fig_fname}")

out = {"sigma0_by_n": sigma0_by_n,
       "mse_by_n": mse_by_n,
       "best_models": best_models}

with open(pkl_fname, "wb") as f:
    pickle.dump(out, f)