"""Generate sigma1 and sigma2 stability notebooks."""
import json, uuid

def cid():
    return str(uuid.uuid4())[:8]

def code(src):
    return {"cell_type":"code","execution_count":None,"id":cid(),"metadata":{},"outputs":[],"source":src}

def md(src):
    return {"cell_type":"markdown","id":cid(),"metadata":{},"source":src}

def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name":"Python 3","language":"python","name":"python3"},
            "language_info": {"name":"python","version":"3.10.0"}
        },
        "nbformat": 4, "nbformat_minor": 5
    }

IMPORTS = """\
import sys, os, yaml, torch, random
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from collections import defaultdict
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from pathlib import Path
from scipy.spatial.distance import pdist
from tqdm.notebook import trange, tqdm

sys.path.append('/orcd/data/jhm/001/om2/jmhicks/projects/TextureStreaming/code/')
sys.path.append('/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/utls/')
sys.path.append('/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/src/model/')
sys.path.append('/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/')

from chexture_choolbox.auditorytexture.texture_model import TextureModel
from chexture_choolbox.auditorytexture.helpers import FlattenStats
from texture_prior.params import model_params, statistics_set, texture_dataset
from texture_prior.utils import path
from utls.plotting import ensure_dir
from utls.loading import (load_results_with_exclusion_2, move_sequences_to_used,
                           load_results_with_exclusion_no_dropping)
from utls.runners_v2 import run_experiment_scores, make_noise_schedule
from utls.runners_utils import *
from encoders import *
from utls.toy_experiments import (
    make_isi_n_block_experiment, make_toy_experiment_list, make_multi_isi_toy_experiments,
)
from utls.sigma_fitting import log_mid, make_grid, auc_to_dprime
"""

LOAD_CFG = """\
def load_config(cfg_path):
    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    with open(cfg_path) as f:
        return yaml.safe_load(f), cfg_path

def median_pairwise_distance(X, metric="euclidean", n_samples=500, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=min(n_samples, X.shape[0]), replace=False)
    return float(np.median(pdist(X[idx], metric=metric)))

CONFIG_PATH = (
    "/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/"
    "model_yamls/three-regime/resnet50/nontime_avg/run_000005.yaml"
)
model_cfg, model_cfg_path = load_config(CONFIG_PATH)
print(model_cfg)
"""

PARSE_DATA = """\
exp_cfg    = model_cfg["experiment"]
which_task = exp_cfg["which_task"]
is_multi   = exp_cfg["is_multi"]
which_isi  = exp_cfg.get("which_isi", None)
isis       = [0, 1, 2, 4, 8, 16, 32, 64] if is_multi else [0, which_isi]
metric     = model_cfg["metric"]
noise_cfg  = model_cfg["noise_model"]
noise_mode = noise_cfg["name"]
t_step     = noise_cfg["t_step"]
repr_cfg   = model_cfg["representation"]
time_avg   = repr_cfg["time_avg"]
encoder_type = repr_cfg["type"]
layer      = repr_cfg.get("layer", None)
pc_dims    = repr_cfg.get("pc_dims", None)

exp_list, all_files, name_to_idx, human_runs, task_name, hr_task_name = (
    load_experiment_data(which_task, which_isi, is_multi, old=False)
)
human_curve  = compute_human_curve(human_runs, is_multi, which_isi)
time_avg_tag = "time_avg" if time_avg else "nontime_avg"
print("ISIs:", isis)
print("Human d':", human_curve)
"""

ENCODER = """\
NN_ENCODERS  = {"kell2018", "resnet50"}
encoder_task = "word_speaker_audioset" if encoder_type in NN_ENCODERS else "audioset"
encoder_cfg  = dict(
    encoder_type=encoder_type, model_name=encoder_type, task=encoder_task,
    statistics_dict=statistics_set.statistics, model_params=model_params,
    pc_dims=pc_dims, sr=20000, duration=2.0, rms_level=0.05,
    time_avg=time_avg, device="cuda",
)
if encoder_type in NN_ENCODERS: encoder_cfg["layer"] = layer
if encoder_type == "texture":   encoder_cfg["pc_dims"] = pc_dims

encoder_name = make_encoder_name(encoder_cfg)
encoder      = build_encoder(encoder_cfg)
X            = encode_stimuli(encoder, all_files)
X_np         = X.detach().cpu().numpy()
print("Shape:", X_np.shape, " NaN?", torch.isnan(X).any().item())

d50 = median_pairwise_distance(X_np, metric="cosine")
print(f"d50 = {d50:.6f}")

param_bounds = {
    "sigma0": (noise_cfg["sigma0_min"],         noise_cfg["sigma0_max"]),
    "sigma1": (noise_cfg["sigma1_min"] * d50,   noise_cfg["sigma1_max"] * d50),
    "sigma2": (noise_cfg["sigma2_min"] * d50,   noise_cfg["sigma2_max"] * d50),
}
for k, v in param_bounds.items():
    print(f"  {k}: ({v[0]:.6f}, {v[1]:.6f})")

stimulus_pool = sorted({s for seq in exp_list for s in seq})
print(f"Stimulus pool: {len(stimulus_pool)}")
"""

# ---------- helpers for the sweep inner-loop ----------
SWEEP_HELPER = """\
def run_sigma_sweep(sigma_name, sigma_grid, fixed_sigmas, exps_by_isi,
                    isi_to_hc_idx, human_curve, N_MC, t_step, noise_mode,
                    metric, X, name_to_idx, base_seed=0):
    \"\"\"Sweep one sigma over sigma_grid; for each value run N_MC reps.
    Returns list of dicts with sigma_value, mse_mean, mse_std, dprime_mean, dprime_std.
    \"\"\"
    results = []
    for sig_idx, sigma_val in enumerate(sigma_grid):
        sigmas = dict(fixed_sigmas)
        sigmas[sigma_name] = sigma_val
        mse_per_rep    = []
        dprime_per_rep = []

        for rep in trange(N_MC, desc=f"{sigma_name}={sigma_val:.4g}", leave=False):
            rep_mse = []
            rep_dp  = []
            for isi_val, exps in exps_by_isi.items():
                if not exps: continue
                hc_idx   = isi_to_hc_idx.get(isi_val)
                if hc_idx is None: continue
                human_dp = human_curve[hc_idx]
                run_out  = run_experiment_scores(
                    sigma0=sigmas["sigma0"], sigma1=sigmas["sigma1"], sigma2=sigmas["sigma2"],
                    t_step=t_step, rate=0, noise_mode=noise_mode,
                    metric=metric, X0=X, name_to_idx=name_to_idx,
                    experiment_list=exps, debug=False,
                    seed=base_seed + isi_val*1_000_000 + sig_idx*10_000 + rep,
                )
                hits = np.asarray(run_out["hits"]); fas = np.asarray(run_out["fas"])
                if len(hits)==0 or len(fas)==0: continue
                y = np.concatenate([np.ones(len(hits)), np.zeros(len(fas))])
                dp = auc_to_dprime(roc_auc_score(y, -np.concatenate([hits,fas])))
                rep_mse.append((dp - human_dp)**2)
                rep_dp.append(dp)
            if rep_mse:
                mse_per_rep.append(np.mean(rep_mse))
                dprime_per_rep.append(np.mean(rep_dp))

        results.append({
            "sigma_value":  sigma_val,
            "mse_mean":     np.mean(mse_per_rep)    if mse_per_rep    else np.nan,
            "mse_std":      np.std(mse_per_rep)     if mse_per_rep    else np.nan,
            "dprime_mean":  np.mean(dprime_per_rep) if dprime_per_rep else np.nan,
            "dprime_std":   np.std(dprime_per_rep)  if dprime_per_rep else np.nan,
        })
    return results
"""

# ======================================================
#  SIGMA-1 NOTEBOOK
# ======================================================
sigma1_cells = [
    md("# Sigma1 Stability Analysis\n\n"
       "Mirrors the sigma0 stability notebook but for **sigma1** (ISIs 1, 2, 4).\n\n"
       "| Section | Question |\n|---------|----------|\n"
       "| A | Per-ISI stability vs n_experiments |\n"
       "| B | Combined-ISI stability |\n"
       "| C | Which ISI subsets are most informative? |"),
    code(IMPORTS),
    code(LOAD_CFG),
    code(PARSE_DATA),
    code(ENCODER),
    code(
        "# ---- fix other sigmas at geometric means ----\n"
        "sigma0_fixed = log_mid(*param_bounds['sigma0'])\n"
        "sigma2_fixed = log_mid(*param_bounds['sigma2'])\n"
        "print(f'Fixed sigma0 = {sigma0_fixed:.6f}')\n"
        "print(f'Fixed sigma2 = {sigma2_fixed:.6f}')\n"
        "\n"
        "isi_to_hc_idx = {isi_val: i for i, isi_val in enumerate(isis)}\n"
        "N_MC      = 64\n"
        "N_PER_DIM = 8\n"
        "K_STIM    = 10   # enough for ISI 1-4 (need >= ISI+1)\n"
        "\n"
        "sigma1_grid = make_grid(param_bounds['sigma1'][0], param_bounds['sigma1'][1],\n"
        "                        N_PER_DIM, spacing='log')\n"
        "print('sigma1 grid:', [f'{v:.5f}' for v in sigma1_grid])\n"
        "\n"
        "print('\\nHuman d\\' targets:')\n"
        "for isi_val in [1, 2, 4]:\n"
        "    idx = isi_to_hc_idx[isi_val]\n"
        "    print(f'  ISI {isi_val}: {human_curve[idx]:.4f}')\n"
    ),
    code(SWEEP_HELPER),
    md("## Section A: Per-ISI Stability\n\n"
       "Each ISI tested independently. Vary n_experiments in [20, 40, 80, 160]."),
    code(
        "EXP_COUNTS = [20, 40, 80, 160]\n"
        "TEST_ISIS  = [1, 2, 4]\n"
        "per_isi_results = {}   # isi -> {n_exp -> list_of_result_dicts}\n"
        "\n"
        "for isi_val in TEST_ISIS:\n"
        "    hc_idx = isi_to_hc_idx[isi_val]\n"
        "    print(f'\\n=== ISI={isi_val}  human d\\' = {human_curve[hc_idx]:.4f} ===')\n"
        "    n_exp_results = {}\n"
        "\n"
        "    for n_exp in EXP_COUNTS:\n"
        "        exps = make_toy_experiment_list(\n"
        "            stimulus_pool, isi=isi_val, n_experiments=n_exp,\n"
        "            k_stimuli=K_STIM, seed=isi_val*1000+n_exp)\n"
        "        print(f'  n_exp={n_exp}: {len(exps)} exps, '\n"
        "              f'avg len {np.mean([len(e) for e in exps]):.1f}')\n"
        "\n"
        "        res = run_sigma_sweep(\n"
        "            sigma_name='sigma1', sigma_grid=sigma1_grid,\n"
        "            fixed_sigmas={'sigma0': sigma0_fixed, 'sigma1': 0, 'sigma2': sigma2_fixed},\n"
        "            exps_by_isi={isi_val: exps},\n"
        "            isi_to_hc_idx=isi_to_hc_idx, human_curve=human_curve,\n"
        "            N_MC=N_MC, t_step=t_step, noise_mode=noise_mode,\n"
        "            metric=metric, X=X, name_to_idx=name_to_idx,\n"
        "            base_seed=isi_val*100_000_000 + n_exp*1_000_000,\n"
        "        )\n"
        "        n_exp_results[n_exp] = res\n"
        "\n"
        "    per_isi_results[isi_val] = n_exp_results\n"
        "\n"
        "print('Done.')\n"
    ),
    code(
        "fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n"
        "for ax, isi_val in zip(axes, TEST_ISIS):\n"
        "    hc_idx = isi_to_hc_idx[isi_val]\n"
        "    for n_exp, res in per_isi_results[isi_val].items():\n"
        "        df = pd.DataFrame(res)\n"
        "        ax.plot(df.sigma_value, df.mse_mean, 'o-', label=f'N={n_exp}')\n"
        "        ax.fill_between(df.sigma_value, df.mse_mean-df.mse_std,\n"
        "                        df.mse_mean+df.mse_std, alpha=0.2)\n"
        "    ax.set_xscale('log')\n"
        "    ax.set_xlabel(r'$\\sigma_1$')\n"
        "    ax.set_ylabel('MSE')\n"
        "    ax.set_title(f'ISI={isi_val}  (human d\\u2032={human_curve[hc_idx]:.2f})')\n"
        "    ax.legend(fontsize=8); ax.grid(alpha=0.3)\n"
        "fig.suptitle(f'Sigma1 per-ISI stability  ({encoder_type}-{layer}-{time_avg_tag})', y=1.03)\n"
        "plt.tight_layout(); plt.show()\n"
        "\n"
        "# d' version\n"
        "fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n"
        "for ax, isi_val in zip(axes, TEST_ISIS):\n"
        "    hc_idx = isi_to_hc_idx[isi_val]\n"
        "    for n_exp, res in per_isi_results[isi_val].items():\n"
        "        df = pd.DataFrame(res)\n"
        "        ax.errorbar(df.sigma_value, df.dprime_mean, yerr=df.dprime_std,\n"
        "                    fmt='o-', capsize=3, label=f'N={n_exp}')\n"
        "    ax.axhline(human_curve[hc_idx], color='k', ls='--', label='Human', alpha=0.6)\n"
        "    ax.set_xscale('log'); ax.set_xlabel(r'$\\sigma_1$'); ax.set_ylabel(\"d'\")\n"
        "    ax.set_title(f'ISI={isi_val}'); ax.legend(fontsize=8); ax.grid(alpha=0.3)\n"
        "plt.tight_layout(); plt.show()\n"
    ),
    md("## Section B: Combined ISI Stability\n\n"
       "All ISIs [1, 2, 4] together — matching Stage B of `three_stage_fit`."),
    code(
        "COMBINED_ISIS      = [1, 2, 4]\n"
        "COMBINED_EXP_COUNTS = [10, 20, 40, 80]\n"
        "combined_results = {}\n"
        "\n"
        "for n_exp in COMBINED_EXP_COUNTS:\n"
        "    print(f'\\n--- n_exp/ISI={n_exp} ---')\n"
        "    exps_by_isi = make_multi_isi_toy_experiments(\n"
        "        stimulus_pool, isi_values=COMBINED_ISIS,\n"
        "        n_experiments_per_isi=n_exp, k_stimuli=K_STIM, seed=42+n_exp)\n"
        "    for iv, el in exps_by_isi.items():\n"
        "        print(f'  ISI {iv}: {len(el)} exps, avg len {np.mean([len(e) for e in el]):.1f}')\n"
        "\n"
        "    res = run_sigma_sweep(\n"
        "        sigma_name='sigma1', sigma_grid=sigma1_grid,\n"
        "        fixed_sigmas={'sigma0': sigma0_fixed, 'sigma1': 0, 'sigma2': sigma2_fixed},\n"
        "        exps_by_isi=exps_by_isi,\n"
        "        isi_to_hc_idx=isi_to_hc_idx, human_curve=human_curve,\n"
        "        N_MC=N_MC, t_step=t_step, noise_mode=noise_mode,\n"
        "        metric=metric, X=X, name_to_idx=name_to_idx,\n"
        "        base_seed=500_000_000 + n_exp*1_000_000,\n"
        "    )\n"
        "    combined_results[n_exp] = res\n"
        "print('Done.')\n"
    ),
    code(
        "plt.figure(figsize=(8, 5))\n"
        "for n_exp, res in combined_results.items():\n"
        "    df = pd.DataFrame(res)\n"
        "    plt.plot(df.sigma_value, df.mse_mean, 'o-', label=f'N exp/ISI={n_exp}')\n"
        "    plt.fill_between(df.sigma_value, df.mse_mean-df.mse_std,\n"
        "                     df.mse_mean+df.mse_std, alpha=0.2)\n"
        "plt.xscale('log'); plt.xlabel(r'$\\sigma_1$'); plt.ylabel('Mean MSE [ISIs 1,2,4]')\n"
        "plt.title(f'Sigma1 combined-ISI stability  N_MC={N_MC}, K={K_STIM}')\n"
        "plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()\n"
    ),
    md("## Section C: Which ISI Subsets Are Most Informative?\n\n"
       "Fixed n_exp=40, compare all subsets of [1,2,4]."),
    code(
        "ISI_SUBSETS = [[1],[2],[4],[1,2],[2,4],[1,4],[1,2,4]]\n"
        "N_EXP_FIXED = 40\n"
        "subset_results = {}\n"
        "\n"
        "for subset in ISI_SUBSETS:\n"
        "    print(f'--- subset={subset} ---')\n"
        "    exps_by_isi = make_multi_isi_toy_experiments(\n"
        "        stimulus_pool, isi_values=subset,\n"
        "        n_experiments_per_isi=N_EXP_FIXED, k_stimuli=K_STIM,\n"
        "        seed=sum(subset)*100+7)\n"
        "    res = run_sigma_sweep(\n"
        "        sigma_name='sigma1', sigma_grid=sigma1_grid,\n"
        "        fixed_sigmas={'sigma0': sigma0_fixed, 'sigma1': 0, 'sigma2': sigma2_fixed},\n"
        "        exps_by_isi=exps_by_isi,\n"
        "        isi_to_hc_idx=isi_to_hc_idx, human_curve=human_curve,\n"
        "        N_MC=N_MC, t_step=t_step, noise_mode=noise_mode,\n"
        "        metric=metric, X=X, name_to_idx=name_to_idx,\n"
        "        base_seed=sum(subset)*100_000_000,\n"
        "    )\n"
        "    subset_results[tuple(subset)] = res\n"
        "print('Done.')\n"
    ),
    code(
        "plt.figure(figsize=(9, 6))\n"
        "for subset_key, res in subset_results.items():\n"
        "    df = pd.DataFrame(res)\n"
        "    plt.plot(df.sigma_value, df.mse_mean, 'o-', label=f'ISIs {list(subset_key)}')\n"
        "    plt.fill_between(df.sigma_value, df.mse_mean-df.mse_std,\n"
        "                     df.mse_mean+df.mse_std, alpha=0.15)\n"
        "plt.xscale('log'); plt.xlabel(r'$\\sigma_1$'); plt.ylabel('Mean MSE')\n"
        "plt.title(f'Which ISI subsets are most informative?  (n={N_EXP_FIXED}/ISI, N_MC={N_MC})')\n"
        "plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()\n"
    ),
    md("## Summary\n\n*(Fill in after running)*\n\n**Recommended Stage B settings:**"),
    code(
        "print('Cost estimates (run_experiment_scores calls per fit):')\n"
        "print(f'  Current: n_grid=15, n_mc=32, n_refine=2, 3 ISIs = {15*32*3*2}')\n"
        "for n_isi in [1,2,3]:\n"
        "    for n_mc in [16,32]:\n"
        "        for n_grid in [10,15]:\n"
        "            print(f'  {n_isi} ISIs n_grid={n_grid} n_mc={n_mc}: {n_grid*n_mc*n_isi*2}')\n"
    ),
]

# ======================================================
#  SIGMA-2 NOTEBOOK
# ======================================================
sigma2_cells = [
    md("# Sigma2 Stability Analysis\n\n"
       "Same approach as sigma1 notebook, but for **sigma2** (ISIs 8, 16, 32, 64).\n\n"
       "**Key challenge:** High ISIs need many stimuli per experiment.\n"
       "`make_isi_n_block_experiment` requires `k >= ISI + 1`:\n"
       "- ISI=8 → k≥9, ISI=16 → k≥17, ISI=32 → k≥33, ISI=64 → k≥65\n\n"
       "| Section | Question |\n|---------|----------|\n"
       "| A | Per-ISI stability vs n_experiments |\n"
       "| B | k_stimuli sensitivity per ISI |\n"
       "| C | Combined-ISI stability |\n"
       "| D | Which ISI subsets are most informative? (Can we drop ISI-64?) |"),
    code(IMPORTS),
    code(LOAD_CFG),
    code(PARSE_DATA),
    code(ENCODER),
    code(
        "sigma0_fixed = log_mid(*param_bounds['sigma0'])\n"
        "sigma1_fixed = log_mid(*param_bounds['sigma1'])\n"
        "print(f'Fixed sigma0 = {sigma0_fixed:.6f}')\n"
        "print(f'Fixed sigma1 = {sigma1_fixed:.6f}')\n"
        "\n"
        "isi_to_hc_idx = {isi_val: i for i, isi_val in enumerate(isis)}\n"
        "N_MC      = 64\n"
        "N_PER_DIM = 8\n"
        "\n"
        "sigma2_grid = make_grid(param_bounds['sigma2'][0], param_bounds['sigma2'][1],\n"
        "                        N_PER_DIM, spacing='log')\n"
        "print('sigma2 grid:', [f'{v:.6f}' for v in sigma2_grid])\n"
        "\n"
        "print('\\nHuman d\\' targets:')\n"
        "for isi_val in [8, 16, 32, 64]:\n"
        "    idx = isi_to_hc_idx.get(isi_val)\n"
        "    if idx is not None:\n"
        "        print(f'  ISI {isi_val}: {human_curve[idx]:.4f}')\n"
        "    else:\n"
        "        print(f'  ISI {isi_val}: NOT in isis list!')\n"
        "\n"
        "print('\\nMin k per ISI (pool size =', len(stimulus_pool), '):')\n"
        "for isi in [8, 16, 32, 64]:\n"
        "    print(f'  ISI {isi}: k >= {isi+1}')\n"
    ),
    code(SWEEP_HELPER),
    md("## Section A: Per-ISI Stability\n\n"
       "Each ISI tested independently. k_stimuli set to `ISI + 2` (minimum for one block).\n"
       "Vary n_experiments in [20, 40, 80, 160]."),
    code(
        "EXP_COUNTS = [20, 40, 80, 160]\n"
        "TEST_ISIS  = [8, 16, 32, 64]\n"
        "per_isi_results = {}\n"
        "\n"
        "for isi_val in TEST_ISIS:\n"
        "    hc_idx = isi_to_hc_idx.get(isi_val)\n"
        "    if hc_idx is None:\n"
        "        print(f'ISI {isi_val} not in data — skip'); continue\n"
        "    k_stim = min(isi_val + 2, len(stimulus_pool))\n"
        "    print(f'\\n=== ISI={isi_val} k={k_stim} human d\\'={human_curve[hc_idx]:.4f} ===')\n"
        "    n_exp_results = {}\n"
        "\n"
        "    for n_exp in EXP_COUNTS:\n"
        "        exps = make_toy_experiment_list(\n"
        "            stimulus_pool, isi=isi_val, n_experiments=n_exp,\n"
        "            k_stimuli=k_stim, seed=isi_val*1000+n_exp)\n"
        "        exps = [e for e in exps if e]\n"
        "        print(f'  n_exp={n_exp}: {len(exps)} exps, '\n"
        "              f'avg len {np.mean([len(e) for e in exps]):.1f}' if exps else f'  n_exp={n_exp}: EMPTY')\n"
        "        if not exps: continue\n"
        "\n"
        "        res = run_sigma_sweep(\n"
        "            sigma_name='sigma2', sigma_grid=sigma2_grid,\n"
        "            fixed_sigmas={'sigma0': sigma0_fixed, 'sigma1': sigma1_fixed, 'sigma2': 0},\n"
        "            exps_by_isi={isi_val: exps},\n"
        "            isi_to_hc_idx=isi_to_hc_idx, human_curve=human_curve,\n"
        "            N_MC=N_MC, t_step=t_step, noise_mode=noise_mode,\n"
        "            metric=metric, X=X, name_to_idx=name_to_idx,\n"
        "            base_seed=isi_val*100_000_000 + n_exp*1_000_000,\n"
        "        )\n"
        "        n_exp_results[n_exp] = res\n"
        "\n"
        "    per_isi_results[isi_val] = n_exp_results\n"
        "print('Done.')\n"
    ),
    code(
        "fig, axes = plt.subplots(1, 4, figsize=(20, 5))\n"
        "for ax, isi_val in zip(axes, TEST_ISIS):\n"
        "    hc_idx = isi_to_hc_idx.get(isi_val)\n"
        "    if hc_idx is None or isi_val not in per_isi_results:\n"
        "        ax.set_title(f'ISI={isi_val} (skip)'); continue\n"
        "    for n_exp, res in per_isi_results[isi_val].items():\n"
        "        df = pd.DataFrame(res)\n"
        "        ax.plot(df.sigma_value, df.mse_mean, 'o-', label=f'N={n_exp}')\n"
        "        ax.fill_between(df.sigma_value, df.mse_mean-df.mse_std,\n"
        "                        df.mse_mean+df.mse_std, alpha=0.2)\n"
        "    ax.set_xscale('log'); ax.set_xlabel(r'$\\sigma_2$'); ax.set_ylabel('MSE')\n"
        "    ax.set_title(f'ISI={isi_val}  d\\'={human_curve[hc_idx]:.2f}')\n"
        "    ax.legend(fontsize=7); ax.grid(alpha=0.3)\n"
        "fig.suptitle(f'Sigma2 per-ISI stability  ({encoder_type}-{layer}-{time_avg_tag})', y=1.03)\n"
        "plt.tight_layout(); plt.show()\n"
    ),
    md("## Section B: k_stimuli Sensitivity\n\n"
       "Fixed n_exp=40. For each ISI, test k = ISI+1, ISI+5, 2*(ISI+1), 3*(ISI+1).\n"
       "Shows how experiment length (more blocks) affects stability."),
    code(
        "N_EXP_FOR_K = 40\n"
        "k_sensitivity = {}   # isi -> {k -> list_of_result_dicts}\n"
        "\n"
        "for isi_val in TEST_ISIS:\n"
        "    hc_idx = isi_to_hc_idx.get(isi_val)\n"
        "    if hc_idx is None: continue\n"
        "\n"
        "    k_min = isi_val + 1\n"
        "    k_candidates = sorted(set(\n"
        "        min(k, len(stimulus_pool))\n"
        "        for k in [k_min, k_min+5, 2*k_min, 3*k_min]\n"
        "    ))\n"
        "    print(f'\\n=== ISI={isi_val}  k candidates: {k_candidates} ===')\n"
        "\n"
        "    k_results = {}\n"
        "    for k_stim in k_candidates:\n"
        "        exps = make_toy_experiment_list(\n"
        "            stimulus_pool, isi=isi_val, n_experiments=N_EXP_FOR_K,\n"
        "            k_stimuli=k_stim, seed=isi_val*500+k_stim)\n"
        "        exps = [e for e in exps if e]\n"
        "        print(f'  k={k_stim}: {len(exps)} exps, '\n"
        "              f'avg len {np.mean([len(e) for e in exps]):.1f}' if exps else f'  k={k_stim}: EMPTY')\n"
        "        if not exps: continue\n"
        "\n"
        "        res = run_sigma_sweep(\n"
        "            sigma_name='sigma2', sigma_grid=sigma2_grid,\n"
        "            fixed_sigmas={'sigma0': sigma0_fixed, 'sigma1': sigma1_fixed, 'sigma2': 0},\n"
        "            exps_by_isi={isi_val: exps},\n"
        "            isi_to_hc_idx=isi_to_hc_idx, human_curve=human_curve,\n"
        "            N_MC=N_MC, t_step=t_step, noise_mode=noise_mode,\n"
        "            metric=metric, X=X, name_to_idx=name_to_idx,\n"
        "            base_seed=isi_val*200_000_000 + k_stim*1_000_000,\n"
        "        )\n"
        "        k_results[k_stim] = res\n"
        "\n"
        "    k_sensitivity[isi_val] = k_results\n"
        "print('Done.')\n"
    ),
    code(
        "fig, axes = plt.subplots(1, 4, figsize=(20, 5))\n"
        "for ax, isi_val in zip(axes, TEST_ISIS):\n"
        "    if isi_val not in k_sensitivity:\n"
        "        ax.set_title(f'ISI={isi_val} (skip)'); continue\n"
        "    for k_stim, res in k_sensitivity[isi_val].items():\n"
        "        df = pd.DataFrame(res)\n"
        "        ax.plot(df.sigma_value, df.mse_mean, 'o-', label=f'k={k_stim}')\n"
        "        ax.fill_between(df.sigma_value, df.mse_mean-df.mse_std,\n"
        "                        df.mse_mean+df.mse_std, alpha=0.15)\n"
        "    ax.set_xscale('log'); ax.set_xlabel(r'$\\sigma_2$'); ax.set_ylabel('MSE')\n"
        "    ax.set_title(f'ISI={isi_val}: k sensitivity (n={N_EXP_FOR_K})')\n"
        "    ax.legend(fontsize=7); ax.grid(alpha=0.3)\n"
        "fig.suptitle(f'Sigma2 k_stimuli sensitivity  ({encoder_type}-{layer}-{time_avg_tag})', y=1.03)\n"
        "plt.tight_layout(); plt.show()\n"
    ),
    md("## Section C: Combined ISI Stability\n\n"
       "All ISIs [8, 16, 32, 64] together. k_stimuli=70 (sufficient for ISI-64)."),
    code(
        "COMBINED_ISIS      = [8, 16, 32, 64]\n"
        "COMBINED_EXP_COUNTS = [10, 20, 40, 80]\n"
        "K_COMBINED = min(70, len(stimulus_pool))\n"
        "print(f'k_combined = {K_COMBINED}')\n"
        "combined_results = {}\n"
        "\n"
        "for n_exp in COMBINED_EXP_COUNTS:\n"
        "    print(f'\\n--- n_exp/ISI={n_exp} ---')\n"
        "    exps_by_isi = make_multi_isi_toy_experiments(\n"
        "        stimulus_pool, isi_values=COMBINED_ISIS,\n"
        "        n_experiments_per_isi=n_exp, k_stimuli=K_COMBINED, seed=200+n_exp)\n"
        "    for iv, el in exps_by_isi.items():\n"
        "        el = [e for e in el if e]; exps_by_isi[iv] = el\n"
        "        print(f'  ISI {iv}: {len(el)} exps, avg len {np.mean([len(e) for e in el]):.1f}' if el else f'  ISI {iv}: EMPTY')\n"
        "\n"
        "    res = run_sigma_sweep(\n"
        "        sigma_name='sigma2', sigma_grid=sigma2_grid,\n"
        "        fixed_sigmas={'sigma0': sigma0_fixed, 'sigma1': sigma1_fixed, 'sigma2': 0},\n"
        "        exps_by_isi={iv: el for iv, el in exps_by_isi.items() if el},\n"
        "        isi_to_hc_idx=isi_to_hc_idx, human_curve=human_curve,\n"
        "        N_MC=N_MC, t_step=t_step, noise_mode=noise_mode,\n"
        "        metric=metric, X=X, name_to_idx=name_to_idx,\n"
        "        base_seed=700_000_000 + n_exp*1_000_000,\n"
        "    )\n"
        "    combined_results[n_exp] = res\n"
        "print('Done.')\n"
    ),
    code(
        "plt.figure(figsize=(8, 5))\n"
        "for n_exp, res in combined_results.items():\n"
        "    df = pd.DataFrame(res)\n"
        "    plt.plot(df.sigma_value, df.mse_mean, 'o-', label=f'N exp/ISI={n_exp}')\n"
        "    plt.fill_between(df.sigma_value, df.mse_mean-df.mse_std,\n"
        "                     df.mse_mean+df.mse_std, alpha=0.2)\n"
        "plt.xscale('log'); plt.xlabel(r'$\\sigma_2$'); plt.ylabel('Mean MSE [ISIs 8,16,32,64]')\n"
        "plt.title(f'Sigma2 combined-ISI stability  N_MC={N_MC}, k={K_COMBINED}')\n"
        "plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()\n"
    ),
    md("## Section D: Which ISI Subsets Are Most Informative?\n\n"
       "Can we drop ISI-64 (needs k≥65, expensive) and still get a stable fit?"),
    code(
        "ISI_SUBSETS = [\n"
        "    [8],[16],[32],[64],\n"
        "    [8,16],[8,32],[8,64],[16,32],[16,64],[32,64],\n"
        "    [8,16,32],[8,16,64],[8,32,64],[16,32,64],\n"
        "    [8,16,32,64],\n"
        "]\n"
        "N_EXP_FIXED = 40\n"
        "subset_results = {}\n"
        "\n"
        "for subset in ISI_SUBSETS:\n"
        "    k_stim = min(max(isi+2 for isi in subset), len(stimulus_pool))\n"
        "    print(f'subset={subset} k={k_stim}')\n"
        "    exps_by_isi = make_multi_isi_toy_experiments(\n"
        "        stimulus_pool, isi_values=subset,\n"
        "        n_experiments_per_isi=N_EXP_FIXED, k_stimuli=k_stim,\n"
        "        seed=sum(subset)*100+13)\n"
        "    exps_by_isi = {iv: [e for e in el if e] for iv, el in exps_by_isi.items()}\n"
        "\n"
        "    res = run_sigma_sweep(\n"
        "        sigma_name='sigma2', sigma_grid=sigma2_grid,\n"
        "        fixed_sigmas={'sigma0': sigma0_fixed, 'sigma1': sigma1_fixed, 'sigma2': 0},\n"
        "        exps_by_isi={iv: el for iv, el in exps_by_isi.items() if el},\n"
        "        isi_to_hc_idx=isi_to_hc_idx, human_curve=human_curve,\n"
        "        N_MC=N_MC, t_step=t_step, noise_mode=noise_mode,\n"
        "        metric=metric, X=X, name_to_idx=name_to_idx,\n"
        "        base_seed=sum(subset)*100_000_000,\n"
        "    )\n"
        "    subset_results[tuple(subset)] = res\n"
        "print('Done.')\n"
    ),
    code(
        "# All subsets\n"
        "plt.figure(figsize=(12, 6))\n"
        "for subset_key, res in subset_results.items():\n"
        "    df = pd.DataFrame(res)\n"
        "    plt.plot(df.sigma_value, df.mse_mean, 'o-',\n"
        "             label=f'{list(subset_key)}', alpha=0.8)\n"
        "plt.xscale('log'); plt.xlabel(r'$\\sigma_2$'); plt.ylabel('Mean MSE')\n"
        "plt.title(f'Which ISI subsets are most informative?  (n={N_EXP_FIXED}/ISI, N_MC={N_MC})')\n"
        "plt.legend(fontsize=6, ncol=3); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()\n"
        "\n"
        "# Single-ISI only\n"
        "plt.figure(figsize=(8, 5))\n"
        "for subset_key, res in subset_results.items():\n"
        "    if len(subset_key) != 1: continue\n"
        "    df = pd.DataFrame(res)\n"
        "    plt.plot(df.sigma_value, df.mse_mean, 'o-', label=f'ISI {subset_key[0]}')\n"
        "    plt.fill_between(df.sigma_value, df.mse_mean-df.mse_std,\n"
        "                     df.mse_mean+df.mse_std, alpha=0.2)\n"
        "plt.xscale('log'); plt.xlabel(r'$\\sigma_2$'); plt.ylabel('MSE')\n"
        "plt.title('Single-ISI contribution to sigma2 MSE')\n"
        "plt.legend(); plt.grid(alpha=0.3); plt.tight_layout(); plt.show()\n"
    ),
    md("## Summary\n\n*(Fill in after running)*\n\n**Recommended Stage C settings:**"),
    code(
        "print('Cost estimates (run_experiment_scores calls per fit):')\n"
        "print(f'  Current: n_grid=15, n_mc=32, n_refine=2, 4 ISIs = {15*32*4*2}')\n"
        "for n_isi in [1,2,3,4]:\n"
        "    for n_mc in [16,32]:\n"
        "        for n_grid in [10,15]:\n"
        "            print(f'  {n_isi} ISIs n_grid={n_grid} n_mc={n_mc}: {n_grid*n_mc*n_isi*2}')\n"
    ),
]

# write notebooks
base = "notebooks"
for fname, cells in [
    ("2026-02-19_sigma1-stability.ipynb", sigma1_cells),
    ("2026-02-19_sigma2-stability.ipynb", sigma2_cells),
]:
    path = f"{base}/{fname}"
    with open(path, "w") as f:
        json.dump(nb(cells), f, indent=1)
    print(f"Written: {path}")
