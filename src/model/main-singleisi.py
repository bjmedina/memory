#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory experiment analysis: min-distance + ROC curves.
Refactored into a single organized script. 
Modify only `run_experiment_at_noise` to explore new dynamics.
"""
# ===================== Imports =====================
import argparse
import sys, os, glob, json, math, datetime
from collections import defaultdict

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LinearRegression

# project-specific paths
sys.path.append('/orcd/data/jhm/001/om2/jmhicks/projects/TextureStreaming/code/')
sys.path.append('../utls/')
sys.path.append('../src/model/')
sys.path.append("/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/")

from chexture_choolbox.auditorytexture.texture_model import TextureModel
from chexture_choolbox.auditorytexture.helpers import FlattenStats
from texture_prior.params import model_params, statistics_set, texture_dataset
from texture_prior.utils import path

import DistanceMemoryModel
import encoders
from utls.plotting import ensure_dir
from utls.loading import load_results_with_exclusion_2, move_sequences_to_used
from utls.runners import run_experiment_scores, run_experiment_scores_itemwise, run_experiment_itemwise_hits_fas
from utls.analysis_helpers import rocs_across_noise, convert_human_to_model_struct, compute_scaling_vs_human, convert_human_to_model_struct
from utls.analysis_helpers import auroc_to_dprime, compute_model_dprime_curve
from utls.analysis_helpers import roc_for_isi, auroc_to_dprime, find_optimal_roc_threshold
from utls.plotting import plot_across_noise, plot_noise_overlays, plot_histograms_all_models, plot_model_grid_summary
from utls.io_utils import make_model_save_dir, save_all_figures, save_single_figure, load_all_runs, save_runs_summary

import os, json
import numpy as np
from scipy.stats import norm
from sklearn.utils import resample
from utls.roc_utils import roc_from_arrays  # if separate, or inline

import math


import numpy as np
import torch
import pandas as pd
from collections import defaultdict

from utls.loading import load_results, load_results_with_isi0_exclusion, load_results_with_isi0_dprime_exclusion, move_sequences_to_used, load_results_with_exclusion
from utls.dprime import recompute_dprime_by_isi_per_subject
from utls.reliability import compute_itemwise_split_half_reliability
from utls.plotting import plot_dprime_by_isi, plot_itemwise_split_half_scatter_df, ensure_dir

from utls.reliability import compute_power_curve
from utls.plotting import plot_power_curve

def plot_groupwise_item_response_scatter(
    results,
    title="Group-wise Item Responses",
    kind="hits",
    seed=42,
    split_method="half",
    save_path=None
):
    """
    Plots a scatter plot of itemwise response rates between two internally split groups.

    Args:
        df (pd.DataFrame): itemwise response matrix (participants x items)
        title (str): plot title
        kind (str): 'hits' or 'false_alarms' for axis labeling
        seed (int): random seed for reproducibility
        split_method (str): 'half' (default) or 'random' to determine splitting method
    """

    #r = results['split_half_reliability'][kind][0]

    df = results['itemwise_responses'][kind]
    import matplotlib.pyplot as plt
    if df.shape[0] < 2:
        raise ValueError("DataFrame must have at least 2 participants to split.")

    np.random.seed(seed)

    n = len(df)
    indices = np.arange(n)

    if split_method == "random":
        np.random.shuffle(indices)

    mid = n // 2
    group1_df = df.iloc[indices[:mid]]
    group2_df = df.iloc[indices[mid:]]

    # Ensure columns are aligned
    common_items = sorted(set(group1_df.columns) & set(group2_df.columns))
    if not common_items:
        raise ValueError("No overlapping items between groups.")

    g1_means = group1_df[common_items].mean(axis=0)
    g2_means = group2_df[common_items].mean(axis=0)

    plt.figure(figsize=(6, 6))
    color = "green" if kind == "hits" else "red"
    plt.scatter(g1_means, g2_means, alpha=0.7, color=color, edgecolor="k")

    # Identity line
    lims = [0, 1]
    plt.plot(lims, lims, "--", color="gray", linewidth=1)

    # Optional: correlation
    #plt.text(0.05, 0.9, f"r = {r:.2f}", transform=plt.gca().transAxes)

    plt.axis('square')
    plt.xlabel(f"Group 1 {kind} rate")
    plt.ylabel(f"Group 2 {kind} rate")
    plt.title(title)
    plt.xlim(lims)
    plt.ylim(lims)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

from scipy.stats import pearsonr, spearmanr
import numpy as np
import pandas as pd


def compute_intergroup_reliability(df_group1, df_group2, method="pearson", min_overlap=10, jitter_std=1e-2):
    """
    Compute intergroup reliability between two groups' itemwise mean responses.

    Adds small bounded jitter if one group's responses are constant.

    Args:
        df_group1, df_group2 : pd.DataFrame
            Rows = participants, columns = item IDs, values = binary or scalar responses.
        method : str
            'pearson' (default) or 'spearman'
        min_overlap : int
            Minimum number of overlapping items to compute a correlation.
        jitter_std : float
            Std dev of jitter to add if one group has zero variance.
    
    Returns:
        (r, n_items, item_df)
    """
    # align items
    common_items = sorted(set(df_group1.columns) & set(df_group2.columns))
    if len(common_items) < min_overlap:
        raise ValueError(f"Not enough overlapping items (found {len(common_items)})")

    # compute itemwise means
    g1_means = df_group1[common_items].mean(axis=0).to_numpy()
    g2_means = df_group2[common_items].mean(axis=0).to_numpy()

    # add bounded jitter if needed
    def add_jitter(vec):
        jitter = np.random.normal(0, jitter_std, size=vec.shape)
        return np.clip(vec + jitter, 0, 1)

    if np.allclose(np.std(g1_means), 0):
        g1_means = add_jitter(g1_means)
    if np.allclose(np.std(g2_means), 0):
        g2_means = add_jitter(g2_means)

    # compute correlation
    if method == "pearson":
        r, _ = pearsonr(g1_means, g2_means)
    elif method == "spearman":
        r, _ = spearmanr(g1_means, g2_means)
    else:
        raise ValueError("method must be 'pearson' or 'spearman'")

    item_df = pd.DataFrame({
        "item": common_items,
        "group1": g1_means,
        "group2": g2_means
    })

    return r, len(common_items), item_df


def main(args):

    # important variables
    PC_DIMS = 256
    DEVICE = 'cuda'
    NV_DEFAULT = 0.1
    save_base=args.save_base
    
    #### GRAB PARTICIPANT FILES ####
    files = glob.glob("/mindhive/mcdermott/www/bjmedina/experiments/mem_exp_v02/results/*V15.csv")
    exps_, seqs_, fnames_ = load_results_with_exclusion_2(
        "/mindhive/mcdermott/www/bjmedina/experiments/mem_exp_v02/results/V15",
        min_dprime=2, min_trials=120, skip_len60=True, verbose=False, return_skipped=False
    )
    
    base_path_ = "/mindhive/mcdermott/www/mturk_stimuli/bjmedina/mem_exp_v15/sequences/"
    
    # Build experiment_list
    experiment_list_ = []
    for seq_ in seqs_:
        with open(base_path_ + seq_, 'r') as f:
            data = json.load(f)
        stim_paths = ["/mindhive/mcdermott/www/mturk_stimuli/bjmedina/mem_exp_v15/" + s 
                      for s in data['filenames_order']]
    
        experiment_list_.append(stim_paths)
    
    # Encode clean reps
    pc_texture_model = encoders.AudioTextureEncoderPCA(
        statistics_dict=statistics_set.statistics,
        pc_dims=PC_DIMS, model_params=model_params,
        sr=20000, rms_level=0.05, duration=2.0, device=DEVICE
    )
        
    # OLD MULTI-ISI DATA
    files = glob.glob("/mindhive/mcdermott/www/bjmedina/experiments/mem_exp_v02/results/*V15.csv")
    exps_, seqs_, fnames_ = load_results_with_exclusion_2(
        "/mindhive/mcdermott/www/bjmedina/experiments/mem_exp_v02/results/V15",
        min_dprime=2, min_trials=256, skip_len60=True, verbose=False, return_skipped=False
    )
    
    base_path_ = "/mindhive/mcdermott/www/mturk_stimuli/bjmedina/mem_exp_v15/sequences/"
    
    # Build experiment_list
    experiment_list_ = []
    for seq_ in seqs_:
        with open(base_path_ + seq_, 'r') as f:
            data = json.load(f)
        stim_paths = ["/mindhive/mcdermott/www/mturk_stimuli/bjmedina/mem_exp_v15/" + s 
                      for s in data['filenames_order']]
    
        experiment_list_.append(stim_paths)
    
    all_files_ = sorted({fn for seq in experiment_list_ for fn in seq})
    name_to_idx_ = {fn: i for i, fn in enumerate(all_files_)}
    
    tmp = DistanceMemoryModel.DistanceMemoryModel(pc_texture_model, NV_DEFAULT, criterion=0.0, device=DEVICE)
    tmp._fill_memory_bank(all_files_)
    with torch.no_grad():
        X0_ = torch.stack([rep.detach().clone().view(-1) for rep in tmp.memory_bank], dim=0).to(DEVICE)

    # SINGLE ISI DATA
    which_task = "atexts-len120"
    
    seqs_paths = {
        "ind-nature-len120": "mem_exp_ind-nature_2025", 
        "global-music-len120": "global-music-2025-n_80",
        "atexts-len120": "mem_exp_atexts_2025",
        "nhs-region-len120": "nhs-region-n_80"
    }
    base_path = "/mindhive/mcdermott/www/mturk_stimuli/bjmedina/{}/sequences/isi_16/len120/"
    exps, seqs, fnames = load_results_with_exclusion_2(
        f"/mindhive/mcdermott/www/bjmedina/experiments/bolivia_2025/results/isi_16/{which_task}",
        min_dprime=2, min_trials=120, skip_len60=True, verbose=False, return_skipped=False
    )
    move_sequences_to_used(base_path.format(seqs_paths[which_task]), seqs)
    
    # Build experiment_list
    experiment_list = []
    for seq in seqs:
        with open(base_path.format(seqs_paths[which_task]) + seq, 'r') as f:
            data = json.load(f)
        stim_paths = ["/mindhive/mcdermott/www/mturk_stimuli/bjmedina/" + seqs_paths[which_task] + "/" + s 
                      for s in data['filenames_order']]
        experiment_list.append(stim_paths)
    
    all_files = sorted({fn for seq in experiment_list for fn in seq})
    name_to_idx = {fn: i for i, fn in enumerate(all_files)}
    
    tmp = DistanceMemoryModel.DistanceMemoryModel(pc_texture_model, NV_DEFAULT, criterion=0.0, device=DEVICE)
    tmp._fill_memory_bank(all_files)
    with torch.no_grad():
        X0 = torch.stack([rep.detach().clone().view(-1) for rep in tmp.memory_bank], dim=0).to(DEVICE)


    human_runs = []
    t_results = glob.glob("/mindhive/mcdermott/www/bjmedina/experiments/mem_exp_v02/results/*V15.csv")

    for result in t_results:
        df = pd.read_csv(result)
        main_exp = df[df['stim_type'] == 'main']
        if len(main_exp) < 256:
            continue
        human_runs.append(convert_human_to_model_struct(main_exp))
    
    # Compute average human d′ vs ISI
    
    isis_human = [0, 1, 2, 3, 4, 8, 16, 32, 64]
    dprimes_human = []
    for k in isis_human:
        aucs = []
        for run in human_runs:
            res = roc_for_isi(run, k)
            if res is not None:
                _, _, auc = res
                aucs.append(auroc_to_dprime(auc))
        dprimes_human.append(np.nanmean(aucs))
        
    human_curve = np.array(dprimes_human, dtype=float)
    human_curve = human_curve[~np.isnan(human_curve)]

    isis = [0, 1, 2, 3, 4, 8, 16, 32, 64]
    human_curve = {
        "isi": np.array(isis_human),
        "dprime": np.array(dprimes_human)
    }

    all_data = load_all_runs(save_base)
    print(f"Loaded {len(all_data)} models.")
    
    best_fits = plot_model_grid_summary(all_data, human_curve, isis, args.save_base, returns=True)
    best_fit_parameters = list(best_fits.keys())

    best_decisions = {}
    
    for key, entry in best_fits.items():
        metric, rate = key
        best_nv  = entry["noise"]
        runs     = entry["runs"]
        run_data = runs[best_nv]
        score_type = run_data.get("score_type", "distance")
    
        hits = np.asarray(run_data["hits"], float)
        fas  = np.asarray(run_data["fas"], float)
        if hits.size == 0 or fas.size == 0:
            continue
    
        roc_info = find_optimal_roc_threshold(hits, fas, score_type=score_type)
    
        if score_type == "distance":
            decision_threshold = -roc_info["threshold"]
        else:
            decision_threshold = roc_info["threshold"]
    
        best_decisions[key] = dict(
            noise=best_nv,
            metric=metric,
            rate=rate,
            encoder="texture_statistics",
            threshold=decision_threshold,
            fpr=roc_info["fpr"],
            tpr=roc_info["tpr"],
            dist=roc_info["distance"],
            rho=entry["fit"]["rho"],
            nmse=entry["fit"]["nmse"],
            score=entry["fit"]["score"],
            drop=entry["fit"]["drop"]
        )
    
        print(f"[{metric} | rate={rate}] "
              f"σ₀={best_nv:g}, R²={entry['fit']['rho']:.3f}, "
              f"thr={decision_threshold:.3f}, FPR={roc_info['fpr']:.3f}, TPR={roc_info['tpr']:.3f}")

    force_rerun = True  # 👈 set to True to recompute and overwrite everything
    
    best_results = {}
    
    for key, best_model in best_decisions.items():
        # --- Parse model info ---
        parsed = best_model["metric"].split("+")
        metric, noise_mode = parsed[0], parsed[1]
        model_name = (
            "DistanceMemoryModel"
            if metric in {"euclidean", "cosine", "mahalanobis", "manhattan"}
            else "LikelihoodMemoryModel"
        )
    
        model_info = dict(
            model_name=model_name,
            metric=metric,
            noise_mode=noise_mode,
            encoder=best_model["encoder"],
            rate=best_model["rate"],
            run_id="prolific_batch"
        )
    
        save_dir = make_model_save_dir(args.save_base, model_info)
        json_path = os.path.join(save_dir, "info_single-isi.json")
    
        # --- Skip if already done ---
        if os.path.exists(json_path) and not force_rerun:
            print(f"⏩ Skipping {metric}+{noise_mode} (rate={best_model['rate']}) — already exists.")
            continue
    
        print(f"🚀 Running model: {metric}+{noise_mode} (rate={best_model['rate']})")
    
        # --- Run the simulation ---
        results_model = run_experiment_itemwise_hits_fas(
            sigma0=best_model["noise"],
            rate=best_model["rate"],
            metric=metric,
            noise_mode=noise_mode,
            X0=X0,
            name_to_idx=name_to_idx,
            experiment_list=experiment_list,
            decision_threshold=best_model["threshold"],
            debug=False,
        )
            
        results = {'itemwise_responses': results_model}
        best_results[key] = results
    
        # --- Save outputs ---
        save_runs_summary(results_model, model_info, save_dir, single_isi=True)


    results_humans = compute_itemwise_split_half_reliability(exps, min_isi=16, max_isi=16)
    hits = results_humans['itemwise_responses']['hits']
    false_alarms  = results_humans['itemwise_responses']['false_alarms']


    import math 
    
    summary = []
    
    for (metric_combo, rate) in best_results:
        entry = best_results[(metric_combo, rate)]
    
        # inside your intergroup reliability loop or preprocessing:
        if "loglikelihood" in metric_combo:
            # flip the sign of responses so higher = stronger "no"
            entry["itemwise_responses"]["hits"] *= -1
            entry["itemwise_responses"]["fas"]  *= -1
    
        r, n_items, item_df = compute_intergroup_reliability(
            hits,
            entry['itemwise_responses']['hits'],
            method="spearman"
        )
    
        if math.isnan(r):
            plot_groupwise_item_response_scatter(
                entry,
                title="Group-wise Item Responses",
                kind="hits",
                seed=42,
                split_method="half",
                save_path=None
            )
            
        r_fa, _, _ = compute_intergroup_reliability(
            false_alarms,
            entry['itemwise_responses']['fas'],
            method="spearman"
            
        )
    
        if math.isnan(r_fa):
            plot_groupwise_item_response_scatter(
                entry,
                title="Group-wise Item Responses",
                kind="fas",
                seed=43,
                split_method="half",
                save_path=None
            )
        
        summary.append({
            "metric_combo": metric_combo,
            "rate": rate,
            "r_hit": r,
            "r_fa": r_fa
        })

    # --- Make DataFrame ---
    df = pd.DataFrame(summary).sort_values(["metric_combo", "rate"])
    
    # --- Plot ---
    plt.figure(figsize=(12, 6))
    x_labels = [f"{m}\nrate={r:g}" for m, r in zip(df["metric_combo"], df["rate"])]
    x = np.arange(len(x_labels))
    
    plt.plot(x, df["r_hit"], 'o-', color='green', label="Hits (Spearman r)")
    plt.plot(x, df["r_fa"],  's--', color='red',   label="False Alarms (Spearman r)")
    
    y_all = np.concatenate([
        df["r_hit"].to_numpy(dtype=float),
        df["r_fa"].to_numpy(dtype=float)
    ])
    y_all = y_all[np.isfinite(y_all)]
    
    if len(y_all) > 0:
        ymin = max(-0.1, np.nanmin(y_all) - 0.05)
        ymax = min(1.05, np.nanmax(y_all) + 0.05)
        plt.ylim(ymin, ymax)
    else:
        plt.ylim(-0.1, 1.05)
    
    
    plt.xticks(x, x_labels, rotation=45, fontsize=5, ha="right")
    plt.ylabel("Intergroup Correlation (Spearman r)")
    plt.title("Model–Human Intergroup Correlation per Metric × Rate")
    plt.ylim(-0.5, 0.5)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.save_base + "model-human_intergroup-correlation.png")
    plt.show()
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_base", type=str, default="/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/figures/model-behavior_v5/")
    args = parser.parse_args()
    main(args)