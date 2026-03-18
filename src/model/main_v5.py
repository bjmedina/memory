"""
main_v5.py — Three-stage sequential sigma fitting via SLURM.

Usage:
    python main_v5.py path/to/run.yaml

Replaces the manual three-stage grid search in main_v4.py with a single
call to ``three_stage_fit()`` from ``utls.sigma_fitting``.

The YAML config is the same format as main_v4, with an optional ``fitting``
section for three_stage_fit hyperparameters (all have sensible defaults).
"""

# ===================== Imports =====================
import matplotlib
matplotlib.use("Agg")  # headless backend for SLURM

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
from utls.runners_v2 import (
    run_experiment_grid,
    run_experiment_scores,
    run_experiment_scores_itemwise,
    run_experiment_itemwise_hits_fas,
    make_noise_schedule,
)
from utls.analysis_helpers import rocs_across_noise, convert_human_to_model_struct, compute_scaling_vs_human
from utls.analysis_helpers import auroc_to_dprime, compute_model_dprime_curve
from utls.analysis_helpers import roc_for_isi
from utls.plotting import plot_across_noise, plot_noise_overlays
from utls.io_utils import make_model_save_dir, save_all_figures, save_single_figure, save_runs_summary
from utls.roc_utils import roc_from_arrays
from utls.runners_utils import *

# three-stage fitting
from utls.sigma_fitting import three_stage_fit, save_three_stage_result


# ===================== Config helpers =====================

def load_config(cfg_path=None):
    """
    Load YAML config.
    Priority:
      1. cfg_path argument (if provided)
      2. sys.argv[1]
    """
    if cfg_path is None:
        if len(sys.argv) != 2:
            raise RuntimeError("Usage: python main_v5.py path/to/run.yaml")
        cfg_path = sys.argv[1]

    cfg_path = Path(cfg_path)
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)

    with open(cfg_path, "r") as f:
        return yaml.safe_load(f), cfg_path


tasks = {
    0: "ind-nature",
    1: "global-music",
    2: "atexts",
}


# ===================== Main =====================

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
    noise_cfg  = model_cfg["noise_model"]
    noise_mode = noise_cfg["name"]
    t_step     = noise_cfg["t_step"]
    noise_tag  = f"{noise_mode}_t{t_step}"

    # ---- representation ----
    repr_cfg     = model_cfg["representation"]
    time_avg     = repr_cfg.get("time_avg", False)
    encoder_type = repr_cfg["type"]
    layer        = repr_cfg.get("layer", None)
    PC_DIMS      = repr_cfg.get("pc_dims", None)

    # ---- fitting hyperparameters (new for v5) ----
    fit_cfg              = model_cfg.get("fitting", {})
    n_grid               = fit_cfg.get("n_grid", 15)
    n_mc                 = fit_cfg.get("n_mc", 32)
    n_refine_iters       = fit_cfg.get("n_refine_iters", 2)
    n_experiments_per_isi = fit_cfg.get("n_experiments_per_isi", 20)
    k_stimuli_per_exp    = fit_cfg.get("k_stimuli_per_exp", 10)

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

    # ---------------------------
    # 3. Output directories
    # ---------------------------
    time_avg_tag = "time_avg" if time_avg else "nontime_avg"
    version = "v13_three-stage"
    save_figs = (
        f"{results_root}/"
        f"figures/"
        f"figures_{version}/"
        f"{task_name}/"
        f"{encoder_type}/"
        f"{time_avg_tag}/"
        f"{metric}/"
        f"{noise_tag}/"
    )

    save_fits     = f"{results_root}/fits/fits_{version}_best"
    save_fits_all = f"{results_root}/fits/fits_{version}_all"

    print("  Figures:", save_figs)
    print("  Fits (all):", save_fits_all)
    print("  Fits (best):", save_fits)

    ensure_dir(save_figs)
    ensure_dir(save_fits)
    ensure_dir(save_fits_all)

    # ---------------------------
    # 4. Build encoder & encode stimuli
    # ---------------------------
    encoder_type = repr_cfg["type"]
    layer        = repr_cfg.get("layer")
    pc_dims      = repr_cfg.get("pc_dims")

    NN_ENCODERS = {"kell2018", "resnet50"}

    encoder_task = (
        "word_speaker_audioset"
        if encoder_type in NN_ENCODERS
        else "audioset"
    )

    encoder_cfg = dict(
        encoder_type=encoder_type,
        model_name=encoder_type,
        task=encoder_task,
        statistics_dict=statistics_set.statistics,
        model_params=model_params,
        sr=20000,
        duration=2.0,
        rms_level=0.05,
        time_avg=time_avg,
        device="cuda",
    )

    # representation-specific fields
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
    X_np = X.detach().cpu().numpy()
    print("Encoded shape:", X_np.shape)

    # ---------------------------
    # 5. Parameter bounds
    # ---------------------------
    param_bounds = {
        "sigma0": (noise_cfg["sigma0_min"], noise_cfg["sigma0_max"]),
        "sigma1": (noise_cfg["sigma1_min"], noise_cfg["sigma1_max"]),
        "sigma2": (noise_cfg["sigma2_min"], noise_cfg["sigma2_max"]),
    }

    print("Parameter bounds:")
    for k, v in param_bounds.items():
        print(f"  {k}: ({v[0]:.6f}, {v[1]:.6f})")

    # ---------------------------
    # 6. Stimulus pool
    # ---------------------------
    stimulus_pool = sorted({s for seq in exp_list for s in seq})
    print(f"Stimulus pool size: {len(stimulus_pool)}")
    assert len(stimulus_pool) >= 65, (
        f"Stimulus pool ({len(stimulus_pool)}) too small for ISI-64 blocks (need >= 65)"
    )

    # ---------------------------
    # 7. Three-stage fit
    # ---------------------------
    print("\n" + "=" * 60)
    print("STARTING THREE-STAGE FIT")
    print("=" * 60)

    fit_result = three_stage_fit(
        run_experiment_fn=run_experiment_scores,
        param_bounds=param_bounds,
        noise_mode=noise_mode,
        metric=metric,
        X0=X,
        name_to_idx=name_to_idx,
        stimulus_pool=stimulus_pool,
        human_curve=human_curve,
        isis=isis,
        t_step=t_step,
        # ISI groupings
        isi_sigma0=[0],
        isi_sigma1=[1, 2, 4],
        isi_sigma2=[8, 16, 32, 64],
        # toy experiment settings
        n_experiments_per_isi=n_experiments_per_isi,
        k_stimuli_per_exp=k_stimuli_per_exp,
        # grid search settings
        n_grid=n_grid,
        n_mc=n_mc,
        n_refine_iters=n_refine_iters,
        seed=0,
        verbose=True,
        plot=False,  # headless on SLURM
    )

    print(f"\nFitted parameters:")
    print(f"  sigma0 = {fit_result['sigma0']:.6f}")
    print(f"  sigma1 = {fit_result['sigma1']:.6f}")
    print(f"  sigma2 = {fit_result['sigma2']:.6f}")

    # ---------------------------
    # 8. Save three-stage result
    # ---------------------------
    fitting_settings = {
        "n_grid": n_grid,
        "n_mc": n_mc,
        "n_refine_iters": n_refine_iters,
        "n_experiments_per_isi": n_experiments_per_isi,
        "k_stimuli_per_exp": k_stimuli_per_exp,
    }

    save_three_stage_result(
        fit_result=fit_result,
        save_dir=save_fits_all,
        config=model_cfg,
        encoder_name=encoder_name,
        task_name=task_name,
        metric=metric,
        noise_mode=noise_mode,
        t_step=t_step,
        human_curve=human_curve,
        isis=isis,
        param_bounds=param_bounds,
        fitting_settings=fitting_settings,
        prefix=f"three_stage_{task_name}_{encoder_name}",
    )

    # ---------------------------
    # 9. Final evaluation on full experiments
    # ---------------------------
    sigma0_fit = fit_result["sigma0"]
    sigma1_fit = fit_result["sigma1"]
    sigma2_fit = fit_result["sigma2"]

    final_pb = {
        "sigma0": (sigma0_fit, sigma0_fit),
        "sigma1": (sigma1_fit, sigma1_fit),
        "sigma2": (sigma2_fit, sigma2_fit),
        "t_step": (t_step, t_step),
    }

    final_result = random_search_gridlike(
        n_samples=1,
        param_bounds=final_pb,
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
        subsample=len(exp_list),
        random_draw=False,
        seed=42,
    )[0]

    print(f"\nFinal evaluation (all ISIs):")
    print(f"  Params: sigma0={sigma0_fit:.4f}, sigma1={sigma1_fit:.4f}, sigma2={sigma2_fit:.4f}")
    print(f"  MSE per ISI: {final_result['mse_per_isi']}")
    if len(final_result['mse_per_isi']) > 1:
        print(f"  MSE (ISI=0):  {final_result['mse_per_isi'][0]:.6f}")
        print(f"  MSE (ISI>=1): {np.mean(final_result['mse_per_isi'][1:]):.6f}")

    # ---------------------------
    # 10. Summary plots
    # ---------------------------
    best_fits = generate_and_plot_model_decay_summary_v5(
        [final_result],
        human_curve,
        isis,
        metric_name="mse_per_isi",
        isi_indices=list(range(len(isis))),
        savedir=save_figs,
        max_rows=1,
        hr_task_name=hr_task_name,
        encoder_name=encoder_name,
    )

    # ---------------------------
    # 11. Save best model
    # ---------------------------
    summary_list_best = save_best_models(
        best_fits, save_fits, prefix=f"{task_name}-{encoder_name}"
    )

    print("\nDone. Results saved to:")
    print(f"  Figures: {save_figs}")
    print(f"  All fits: {save_fits_all}")
    print(f"  Best fits: {save_fits}")
