# /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/utls/main.py
import argparse, numpy as np, torch
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory experiment analysis: min-distance + ROC curves.
Refactored into a single organized script. 
Modify only `run_experiment_at_noise` to explore new dynamics.
"""

# ===================== Imports =====================
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
from utls.runners import run_experiment_scores
from utls.analysis_helpers import rocs_across_noise, convert_human_to_model_struct, compute_scaling_vs_human, convert_human_to_model_struct
from utls.analysis_helpers import auroc_to_dprime, compute_model_dprime_curve
from utls.analysis_helpers import roc_for_isi, auroc_to_dprime
from utls.plotting import plot_across_noise, plot_noise_overlays
from utls.io_utils import make_model_save_dir, save_all_figures, save_single_figure, save_runs_summary

import os, json
import numpy as np
from scipy.stats import norm
from sklearn.utils import resample
from utls.roc_utils import roc_from_arrays 

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
        
        all_files_ = sorted({fn for seq in experiment_list_ for fn in seq})
        
        tmp = DistanceMemoryModel.DistanceMemoryModel(pc_texture_model, NV_DEFAULT, criterion=0.0, device=DEVICE)
        
        tmp._fill_memory_bank(all_files_)
        
        with torch.no_grad():
            X0_ = torch.stack([rep.detach().clone().view(-1) for rep in tmp.memory_bank], dim=0).to(DEVICE)
        

        # get human data
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
        
        pc_texture_model = encoders.AudioTextureEncoderPCA(
            statistics_dict=statistics_set.statistics,
            pc_dims=PC_DIMS, model_params=model_params,
            sr=20000, rms_level=0.05, duration=2.0, device=DEVICE
        )
        
        all_files_ = sorted({fn for seq in experiment_list_ for fn in seq})
        
        tmp = DistanceMemoryModel.DistanceMemoryModel(pc_texture_model, NV_DEFAULT, criterion=0.0, device=DEVICE)
        tmp._fill_memory_bank(all_files_)
        with torch.no_grad():
            X0_ = torch.stack([rep.detach().clone().view(-1) for rep in tmp.memory_bank], dim=0).to(DEVICE)
            
        name_to_idx_ = {fn: i for i, fn in enumerate(all_files_)}
                        
        # Determine model type from metric
        model_name = "DistanceMemoryModel" if args.metric in {"euclidean", "cosine", "mahalanobis", "manhattan"} else "LikelihoodMemoryModel"


        RATES = np.geomspace(0.01, 5, 10)   # or whatever values you want

        rate_ = RATES[args.rate]

        print(f"\n=== Running model at rate={rate_:.4f} ===")


        model_info = dict(
            model_name=model_name,
            metric=args.metric,
            noise_mode=args.noise_mode,
            encoder=args.encoder,
            rate=rate_,
            run_id=args.run_id
        )
        
        # Define your noise levels (same for all)
        NOISE_LEVELS = np.geomspace(0.01, 5, 10)
        
        curves, runs = rocs_across_noise(
                NOISE_LEVELS,
                rate=rate_,
                runner=run_experiment_scores,
                X0=X0_, name_to_idx=name_to_idx_, experiment_list=experiment_list_,
                metric=args.metric,
                noise_mode=args.noise_mode
        )
        
        save_dir = make_model_save_dir(args.save_base, model_info, which_task=which_task)
        plot_across_noise(NOISE_LEVELS, 
                          runs,  
                          isis=(1,2,3,4,9,17,33,65),
                          model_info=model_info, 
                          save_base=args.save_base)
        plot_noise_overlays(curves, save_dir=save_dir)

        save_runs_summary(runs, model_info, save_dir)


        # # UNIVERSALITY analysis (Across all noise levels)
        # scaling_results = compute_scaling_vs_human(runs, NOISE_LEVELS, human_curve)
        # r2s = [scaling_results[nv]['r2'] for nv in NOISE_LEVELS]
        
        # plt.figure(figsize=(7,5))
        # plt.plot(NOISE_LEVELS, r2s, 'o--', color='orange', label='$R^2$ fit quality')
        
        # plt.ylim([min(np.min(r2s), 0)-0.01,1])
        # plt.xscale('log')
        # plt.xlabel("Noise level (σ₀)")
        # plt.ylabel("$R^2$ (fit quality)")
        # plt.title("Model–Human Universality Fit Across Noise Levels")
        # plt.grid(True, alpha=0.3)
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(f"{save_dir}/Universality_fit.png")
        # plt.show()
        # plt.close()

        # # UNIVERSALITY analysis (Across all noise levels)
        # # for the next noise level
        # y_ref    = human_curve
        # isi_ref = [0, 1, 2, 3, 4, 8, 16, 32, 64]
        
        # best_idx = int(np.nanargmax(r2s))
        # best_nv  = NOISE_LEVELS[best_idx]
        # best_fit = scaling_results[best_nv]
        # print(f"Best σ₀ = {best_nv:.3g}  (α={best_fit['alpha']:.3f}, β={best_fit['beta']:.3f}, R²={best_fit['r2']:.3f})")
        
        # # Compute model curve
        # _, y_model = compute_model_dprime_curve(runs[best_nv])
        # y_scaled = best_fit['alpha'] * y_model + best_fit['beta'] 
        # print(y_model)# scale model to human space
        
        # plt.figure(figsize=(7,5))
        # plt.errorbar(isi_ref, y_ref, fmt='o-', color='black', label='Human data')
        # plt.plot(isi_ref, y_model, 's--', color='gray', alpha=0.5, label=f'Model (σ₀={best_nv:g})')
        # plt.plot(isi_ref, y_scaled, 'o--', color='orange', label=f'Model scaled → Human (α={best_fit["alpha"]:.2f}, β={best_fit["beta"]:.2f})')
        
        # plt.xlabel("ISI")
        # plt.ylabel("d′ (z(AUROC))")
        # plt.title(f"Best universality fit (σ₀={best_nv:g}, R²={best_fit['r2']:.2f})")
        # plt.grid(True, alpha=0.3)
        # plt.legend()
        # plt.tight_layout()
        # plt.savefig(f"{save_dir}/best_universality_fit.png")
        # plt.show()
        # plt.close()
        
        print(f"✅ Completed: {model_info}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--noise_mode", type=str, required=True)
    parser.add_argument("--encoder", type=str, default="texture_statistics")
    parser.add_argument("--rate", type=int, default=0.4)
    parser.add_argument("--run_id", type=str, default="prolific_mem_exp_V15_version1")
    parser.add_argument("--save_base", type=str, default="/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/figures/model-behavior_v5/")
    args = parser.parse_args()
    main(args)