"""
main_v6.py — Three-stage sequential sigma fitting (compact stages B/C) via SLURM.

Usage:
    python main_v6.py path/to/run.yaml

Diff vs main_v5.py:
- Stage A (sigma0): per-ISI toy experiments at ISI=0.
- Stage B (sigma1): compact multi-ISI sequences at ISIs [1,2,4].
- Stage C (sigma2): compact multi-ISI sequences at ISIs [8,16,32,64].

This mirrors notebooks/2026-02-26_three-stage-compact-fitting.ipynb while
keeping output artifacts compatible with downstream analysis notebooks.
"""

# ===================== Imports =====================
import matplotlib
matplotlib.use("Agg")

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

from utls.sigma_fitting import fit_sigma_1d, log_mid, save_three_stage_result
from utls.toy_experiments import make_toy_experiment_list, make_compact_multi_isi_sequences


def load_config(cfg_path=None):
    if cfg_path is None:
        if len(sys.argv) != 2:
            raise RuntimeError("Usage: python main_v6.py path/to/run.yaml")
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


if __name__ == "__main__":

    model_cfg, model_cfg_path = load_config()

    print("Running config:")
    print("  YAML:", model_cfg_path)
    print("  Encoder:", model_cfg["representation"]["type"])
    print("  Metric:", model_cfg["metric"])
    print("  Noise:", model_cfg["noise_model"]["name"])

    exp_cfg = model_cfg["experiment"]
    which_task = exp_cfg["which_task"]
    is_multi = exp_cfg["is_multi"]
    which_isi = exp_cfg.get("which_isi", None)

    if is_multi:
        isis = [0, 1, 2, 4, 8, 16, 32, 64]
    else:
        assert which_isi is not None, "which_isi required if not multi-ISI"
        isis = [0, which_isi]

    metric = model_cfg["metric"]

    noise_cfg = model_cfg["noise_model"]
    noise_mode = noise_cfg["name"]
    t_step = noise_cfg["t_step"]
    noise_tag = f"{noise_mode}_t{t_step}"

    repr_cfg = model_cfg["representation"]
    time_avg = repr_cfg.get("time_avg", False)
    encoder_type = repr_cfg["type"]
    layer = repr_cfg.get("layer", None)
    pc_dims = repr_cfg.get("pc_dims", None)

    fit_cfg = model_cfg.get("fitting", {})
    n_grid = fit_cfg.get("n_grid", 15)
    n_mc = fit_cfg.get("n_mc", 32)
    n_refine_iters = fit_cfg.get("n_refine_iters", 2)
    n_experiments_per_isi = fit_cfg.get("n_experiments_per_isi", 20)
    k_stimuli_per_exp = fit_cfg.get("k_stimuli_per_exp", 10)

    compact_cfg = model_cfg.get("compact_fitting", {})
    sigma1_isis = compact_cfg.get("sigma1_isis", [1, 2, 4])
    sigma1_length = compact_cfg.get("sigma1_length", 60)
    sigma1_n_seqs = compact_cfg.get("sigma1_n_seqs", 30)
    sigma1_min_pairs = compact_cfg.get("sigma1_min_pairs", 5)

    sigma2_isis = compact_cfg.get("sigma2_isis", [8, 16, 32, 64])
    sigma2_length = compact_cfg.get("sigma2_length", 75)
    sigma2_n_seqs = compact_cfg.get("sigma2_n_seqs", 30)
    sigma2_min_pairs = compact_cfg.get("sigma2_min_pairs", 5)

    n_seqs_per_rep = compact_cfg.get("n_seqs_per_rep", 10)

    exp_list, all_files, name_to_idx, human_runs, task_name, hr_task_name = load_experiment_data(
        which_task, which_isi, is_multi)

    human_curve = compute_human_curve(human_runs, is_multi, which_isi)
    human_dprimes_by_isi = {int(isi): float(dp) for isi, dp in zip(isis, human_curve)}

    results_root = model_cfg["results_root"]

    time_avg_tag = "time_avg" if time_avg else "nontime_avg"
    version = "v15_three-stage-compact"
    save_figs = (
        f"{results_root}/figures/figures_{version}/{task_name}/{encoder_type}/{time_avg_tag}/{metric}/{noise_tag}/"
    )

    save_fits = f"{results_root}/fits/fits_{version}_best"
    save_fits_all = f"{results_root}/fits/fits_{version}_all"

    ensure_dir(save_figs)
    ensure_dir(save_fits)
    ensure_dir(save_fits_all)

    NN_ENCODERS = {"kell2018", "resnet50"}
    encoder_task = "word_speaker_audioset" if encoder_type in NN_ENCODERS else "audioset"

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

    param_bounds = {
        "sigma0": (noise_cfg["sigma0_min"], noise_cfg["sigma0_max"]),
        "sigma1": (noise_cfg["sigma1_min"], noise_cfg["sigma1_max"]),
        "sigma2": (noise_cfg["sigma2_min"], noise_cfg["sigma2_max"]),
    }

    stimulus_pool = sorted({s for seq in exp_list for s in seq})

    print("\n" + "=" * 60)
    print("STARTING THREE-STAGE FIT (COMPACT B/C)")
    print("=" * 60)

    # Stage A: sigma0 with ISI=0 toy experiments
    stage_a_experiments = {
        0: make_toy_experiment_list(
            stimulus_pool,
            isi=0,
            n_experiments=n_experiments_per_isi,
            k_stimuli=k_stimuli_per_exp,
            seed=101,
        )
    }

    stage_a = fit_sigma_1d(
        run_experiment_fn=run_experiment_scores,
        sigma_name="sigma0",
        sigma_bounds=param_bounds["sigma0"],
        fixed_sigmas={"sigma0": log_mid(*param_bounds["sigma0"]), "sigma1": log_mid(*param_bounds["sigma1"]), "sigma2": log_mid(*param_bounds["sigma2"])},
        noise_mode=noise_mode,
        metric=metric,
        X0=X,
        name_to_idx=name_to_idx,
        experiments_by_isi=stage_a_experiments,
        human_dprimes_by_isi=human_dprimes_by_isi,
        t_step=t_step,
        n_grid=n_grid,
        n_mc=n_mc,
        n_refine_iters=n_refine_iters,
        spacing="log",
        seed=1000,
        verbose=True,
    )
    sigma0_fit = stage_a["best_sigma"]

    # Stage B: sigma1 with compact multi-ISI sequences (1,2,4)
    stage_b_experiment_list, stage_b_isi_keys = make_compact_multi_isi_sequences(
        stimulus_pool=stimulus_pool,
        isi_values=sigma1_isis,
        n_sequences=sigma1_n_seqs,
        length=sigma1_length,
        min_pairs_per_isi=sigma1_min_pairs,
        seed=202,
    )

    stage_b = fit_sigma_1d(
        run_experiment_fn=run_experiment_scores,
        sigma_name="sigma1",
        sigma_bounds=param_bounds["sigma1"],
        fixed_sigmas={"sigma0": sigma0_fit, "sigma1": log_mid(*param_bounds["sigma1"]), "sigma2": log_mid(*param_bounds["sigma2"])},
        noise_mode=noise_mode,
        metric=metric,
        X0=X,
        name_to_idx=name_to_idx,
        human_dprimes_by_isi=human_dprimes_by_isi,
        t_step=t_step,
        n_grid=n_grid,
        n_mc=n_mc,
        n_refine_iters=n_refine_iters,
        spacing="log",
        seed=2000,
        verbose=True,
        experiment_list=stage_b_experiment_list,
        isi_keys=stage_b_isi_keys,
        target_isis=sigma1_isis,
        n_seqs_per_rep=n_seqs_per_rep,
    )
    sigma1_fit = stage_b["best_sigma"]

    # Stage C: sigma2 with compact multi-ISI sequences (8,16,32,64)
    stage_c_experiment_list, stage_c_isi_keys = make_compact_multi_isi_sequences(
        stimulus_pool=stimulus_pool,
        isi_values=sigma2_isis,
        n_sequences=sigma2_n_seqs,
        length=sigma2_length,
        min_pairs_per_isi=sigma2_min_pairs,
        seed=303,
    )

    stage_c = fit_sigma_1d(
        run_experiment_fn=run_experiment_scores,
        sigma_name="sigma2",
        sigma_bounds=param_bounds["sigma2"],
        fixed_sigmas={"sigma0": sigma0_fit, "sigma1": sigma1_fit, "sigma2": log_mid(*param_bounds["sigma2"])},
        noise_mode=noise_mode,
        metric=metric,
        X0=X,
        name_to_idx=name_to_idx,
        human_dprimes_by_isi=human_dprimes_by_isi,
        t_step=t_step,
        n_grid=n_grid,
        n_mc=n_mc,
        n_refine_iters=n_refine_iters,
        spacing="log",
        seed=3000,
        verbose=True,
        experiment_list=stage_c_experiment_list,
        isi_keys=stage_c_isi_keys,
        target_isis=sigma2_isis,
        n_seqs_per_rep=n_seqs_per_rep,
    )
    sigma2_fit = stage_c["best_sigma"]

    fit_result = {
        "sigma0": sigma0_fit,
        "sigma1": sigma1_fit,
        "sigma2": sigma2_fit,
        "stage_a": stage_a,
        "stage_b": stage_b,
        "stage_c": stage_c,
    }

    fitting_settings = {
        "n_grid": n_grid,
        "n_mc": n_mc,
        "n_refine_iters": n_refine_iters,
        "n_experiments_per_isi": n_experiments_per_isi,
        "k_stimuli_per_exp": k_stimuli_per_exp,
        "compact_fitting": {
            "sigma1_isis": sigma1_isis,
            "sigma1_length": sigma1_length,
            "sigma1_n_seqs": sigma1_n_seqs,
            "sigma1_min_pairs": sigma1_min_pairs,
            "sigma2_isis": sigma2_isis,
            "sigma2_length": sigma2_length,
            "sigma2_n_seqs": sigma2_n_seqs,
            "sigma2_min_pairs": sigma2_min_pairs,
            "n_seqs_per_rep": n_seqs_per_rep,
        },
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
        prefix=f"three_stage_compact_{task_name}_{encoder_name}",
    )

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

    best_fits = generate_and_plot_compact_summary(
        final_result,
        human_curve,
        isis,
        t_step=t_step,
        savedir=save_figs,
        hr_task_name=hr_task_name,
        encoder_name=encoder_name,
    )

    save_best_models(best_fits, save_fits, prefix=f"{task_name}-{encoder_name}")

    print("\nDone. Results saved to:")
    print(f"  Figures: {save_figs}")
    print(f"  All fits: {save_fits_all}")
    print(f"  Best fits: {save_fits}")
