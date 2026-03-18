import sys
import glob
import torch

sys.path.append('../utils/')
sys.path.append('../src/model/')

sys.path.append("/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/")

from utils.plotting import plot_dprime_by_isi, plot_itemwise_split_half_scatter_df, ensure_dir
from utils.dprime import recompute_dprime_by_isi_per_subject
from utils.reliability import compute_itemwise_split_half_reliability

import NoisyAgeMixtureMemoryModel
import encoders

sys.path.append('/orcd/data/jhm/001/om2/bjmedina/')

from chexture_choolbox.auditorytexture.statistics_sets import (
    STAT_SET_FULL_MCDERMOTTSIMONCELLI as statistics_dict
)
from chexture_choolbox.auditorytexture.texture_model import TextureModel
from chexture_choolbox.auditorytexture.helpers import FlattenStats
from texture_prior.params import model_params

import json
import glob
import sys
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from scipy.stats import norm
from collections import defaultdict

import numpy as np
from typing import Callable, List, Dict

from utils.loading import load_results, load_results_with_isi0_exclusion, load_results_with_isi0_dprime_exclusion, move_sequences_to_used, load_results_with_exclusion

# from typing import Dict, List
import numpy as np
from dprime import recompute_dprime_by_isi_per_subject
from NoisyAgeMixtureMemoryModel import NoisyAgeMixtureMemoryModel  # adjust import if needed
from scipy.optimize import minimize

import random 
evaluation_log = []
result_log = []
inits = []

from scipy.optimize import basinhopping

def evaluate_model_dprime_fit(
    memory_model,
    param_dict: Dict[str, float],
    encoding_model,
    experiment_list: List[Dict[str, List[str]]],
    human_sensitivity: np.ndarray,
    device: str = "cpu",
    verbose: bool = False,
) -> float:
    """
    Evaluate model fit to human data by computing MSE between model and human d-primes.

    Args:
        param_dict (dict): Parameters for the memory model.
        encoding_model: A callable encoder.
        experiment_list (list): Each item is a dict with 'stimuli' and 'yt_ids' (same length).
        human_sensitivity (np.ndarray): Ground truth human d-primes, per ISI.
        device (str): Device string.
        verbose (bool): Print status.

    Returns:
        float: Mean squared error between model and human d-primes.
    """
    all_dfs = []

    for i, exp in enumerate(experiment_list):
        model = memory_model(
            encoding_model=encoding_model,
            noise_variance=param_dict.get("noise_variance", 1.0),
            criterion=param_dict.get("criterion", 0.5),
            device=device
        )
        df = model.do_experiment(exp, yt_ids=None, verbose=False)
        all_dfs.append(df)

        if verbose and i % 1 == 0:
            print(f"Finished model simulation for experiment {i}")

    dprime_df = recompute_dprime_by_isi_per_subject(all_dfs, criterion=0)
    model_dprimes = get_dprime_by_isi(dprime_df)  # shape must match human_sensitivity

    if len(model_dprimes) != len(human_sensitivity):
        raise ValueError("Mismatch between model and human d-prime lengths.")

    mse = np.mean((np.array(model_dprimes) - np.array(human_sensitivity)) ** 2)

    if verbose:
        print(f"Model d′: {np.round(model_dprimes, 3)}")
        print(f"Human d′: {np.round(human_sensitivity, 3)}")
        print(f"MSE: {mse:.4f}")

    return mse


def compute_dprime(hit_rate, fa_rate):
    hit_rate = np.clip(hit_rate, 0.0001, 1 - 0.0001)
    fa_rate = np.clip(fa_rate, 0.0001, 1 - 0.0001)
    return norm.ppf(hit_rate) - norm.ppf(fa_rate)


def recompute_dprime_by_isi(exps, criterion=1):
    hit_counts = defaultdict(int)
    fa_counts  = defaultdict(int)
    signal_counts = defaultdict(int)
    noise_counts  = defaultdict(int)

    for df in exps:
        seen_yt_ids = {}
        yt_ids = df['yt_id'].tolist()
        responses = df['response'].tolist()
        repeats = df['repeat'].tolist()


        for i, (yt, resp, repeat) in enumerate(zip(yt_ids, responses, repeats )):
            if pd.isna(resp) or pd.isna(yt):
                continue

            is_yes = int(int(resp) > criterion)

            if repeat == 'true':
                j = yt_ids[:i].index(yt)
                isi = i - j - 1

                if isi not in [-1,0,1,2,3, 4, 8, 16, 32, 64]:
                    continue

                #print(j, i, isi)
                #print(yt_ids[j], yt)
                signal_counts[isi] += 1
                hit_counts[isi] += is_yes
            elif repeat == 'false':
                noise_counts[-1] += 1  # ISI=-1 for noise trials
                fa_counts[-1] += is_yes

                #seen_yt_ids[yt] = i  # store first appearance

        #print("--")

    # Build results
    results = []
    all_isi_vals = sorted(set(signal_counts) | set(noise_counts))
    for isi in all_isi_vals:
        hits = hit_counts[isi]
        fas  = fa_counts[-1]
        n_signal = signal_counts[isi]
        n_noise  = noise_counts[-1]

        hit_rate = hits / n_signal if n_signal > 0 else np.nan
        fa_rate  = fas  / n_noise  if n_noise  > 0 else np.nan
        dprime_val = compute_dprime(hit_rate, fa_rate) #if np.isfinite(hit_rate) and np.isfinite(fa_rate) else np.nan

        results.append({
            'isi': isi,
            'hits': hits,
            'false_alarms': fas,
            'n_signal': n_signal,
            'n_noise': n_noise,
            'hit_rate': hit_rate,
            'fa_rate': fa_rate,
            'd_prime': dprime_val
        })

    return pd.DataFrame(results).sort_values(by='isi')


def recompute_dprime_by_isi_per_subject(exps, criterion=1):
    allowed_isi = {-1, 0, 1, 2, 3, 4, 8, 16, 32, 64}
    all_results = []

    for subj_idx, df in enumerate(exps):
        hit_counts = defaultdict(int)
        fa_counts  = defaultdict(int)
        signal_counts = defaultdict(int)
        noise_counts  = defaultdict(int)

        yt_ids = df['yt_id'].tolist()
        responses = df['response'].tolist()
        repeats = df['repeat'].tolist()

        for i, (yt, resp, repeat) in enumerate(zip(yt_ids, responses, repeats)):
            if pd.isna(resp) or pd.isna(yt):
                continue

            is_yes = int(int(resp) > criterion)

            if repeat == 'true':
                try:
                    j = yt_ids[:i].index(yt)
                    isi = i - j - 1
                except ValueError:
                    continue  # yt not found in earlier trials

                if isi not in allowed_isi:
                    continue

                signal_counts[isi] += 1
                hit_counts[isi] += is_yes

            elif repeat == 'false':
                isi = -1
                noise_counts[isi] += 1
                fa_counts[isi] += is_yes

        # Aggregate per-ISI results for this subject
        for isi in sorted(signal_counts.keys() | noise_counts.keys()):
            n_signal = signal_counts[isi]
            n_noise  = noise_counts.get(-1, 0)  # all noise trials pooled under -1
            hits = hit_counts[isi]
            fas  = fa_counts.get(-1, 0)

            hit_rate = hits / n_signal if n_signal > 0 else np.nan
            fa_rate  = fas  / n_noise  if n_noise  > 0 else np.nan
            d_prime = compute_dprime(hit_rate, fa_rate) if np.isfinite(hit_rate) and np.isfinite(fa_rate) else np.nan

            all_results.append({
                'subject': subj_idx,
                'isi': isi,
                'hits': hits,
                'false_alarms': fas,
                'n_signal': n_signal,
                'n_noise': n_noise,
                'hit_rate': hit_rate,
                'fa_rate': fa_rate,
                'd_prime': d_prime
            })

    return pd.DataFrame(all_results).sort_values(by=['subject', 'isi'])


def load_results(results_dir, isi_pow=2, min_trials=120, skip_len60=True):
    """Load and filter experiment result CSVs."""
    files = sorted(
        [f for f in os.listdir(results_dir) if f.endswith(".csv")],
        key=lambda fn: os.path.getctime(os.path.join(results_dir, fn))
    )
    exps, seqs, fnames = [], [], []
    for fn in files:
        df = pd.read_csv(os.path.join(results_dir, fn))
        main = df[df.stim_type == "main"]
        seq_file = main.sequence_file.iloc[0].split("/")[-1]
        if len(main) < min_trials: continue
        if "tol0" in seq_file: continue
        exps.append(main); seqs.append(seq_file); fnames.append(fn)
    return exps, seqs, fnames

def remove_sequences_with_len60(seq_dir):
    """Remove entries containing 'len60' from unused.json and used.json."""
    for sub in ("unused", "used"):
        path = os.path.join(seq_dir, sub, f"{sub}.json")
        data = json.load(open(path))
        filtered = [f for f in data if "len60" not in f]
        json.dump(sorted(filtered), open(path, "w"), indent=2)

def move_sequences_to_used(seq_dir, seqs_used):
    """Move used sequence filenames from unused.json to used.json."""
    u_path = os.path.join(seq_dir, "unused", "unused.json")
    z_path = os.path.join(seq_dir, "used",   "used.json")
    unused = json.load(open(u_path)); used = json.load(open(z_path))
    seqs = [os.path.basename(s) for s in seqs_used]
    new_unused = [s for s in unused if s not in seqs]
    new_used = sorted(set(used + seqs))
    json.dump(sorted(new_unused), open(u_path, "w"), indent=2)
    json.dump(new_used,         open(z_path, "w"), indent=2)

def get_dprime_by_isi(df_per_subject, return_sem=False, return_subjects=False):
    """
    Compute mean d-prime per ISI across subjects, excluding ISI = -1 (lures).

    Args:
        df_per_subject (pd.DataFrame): Output from recompute_dprime_by_isi_per_subject.
        return_sem (bool): Whether to return standard error of the mean.
        return_subjects (bool): Whether to return per-subject d-primes too.

    Returns:
        pd.DataFrame or dict:
            If return_sem=False:
                DataFrame with columns ['isi', 'd_prime']
            If return_sem=True:
                DataFrame with columns ['isi', 'd_prime', 'sem']
            If return_subjects=True:
                Returns a dict with:
                    'summary': summary DataFrame as above,
                    'per_subject': filtered per-subject df
    """
    df_filtered = df_per_subject[df_per_subject["isi"] > -1]

    grouped = df_filtered.groupby("isi")["d_prime"]
    result_df = grouped.mean().reset_index(name="d_prime")

    if return_sem:
        result_df["sem"] = grouped.sem().values

    if return_subjects:
        return {
            "summary": result_df,
            "per_subject": df_filtered.copy()
        }

    return result_df.d_prime.tolist()

import numpy as np
from dprime import recompute_dprime_by_isi_per_subject
from NoisyAgeMixtureMemoryModel import NoisyAgeMixtureMemoryModel  # adjust import if needed
from scipy.optimize import minimize


# def __init__(self, encoding_model, noise_slope=1.0, noise_offset=1e-3, criterion=0.5, device='cpu'):
#     super(MixtureMemoryModel, self).__init__()
#     self.encoding_model = encoding_model
#     self.noise_slope = noise_slope
#     self.noise_offset = noise_offset
#     self.criterion = criterion
#     self.device = device
#     self.memory_bank = []
#     self.debug_mode = False
    
def evaluate_model_dprime_fit(
    memory_model,
    param_dict: Dict[str, float],
    encoding_model,
    experiment_list: List[Dict[str, List[str]]],
    human_sensitivity: np.ndarray,
    device: str = "cpu",
    verbose: bool = False,
) -> float:
    """
    Evaluate model fit to human data by computing MSE between model and human d-primes.

    Args:
        param_dict (dict): Parameters for the memory model.
        encoding_model: A callable encoder.
        experiment_list (list): Each item is a dict with 'stimuli' and 'yt_ids' (same length).
        human_sensitivity (np.ndarray): Ground truth human d-primes, per ISI.
        device (str): Device string.
        verbose (bool): Print status.

    Returns:
        float: Mean squared error between model and human d-primes.
    """
    all_dfs = []

    for i, exp in enumerate(experiment_list):
        model = memory_model(
            encoding_model=encoding_model,
            noise_slope=param_dict.get("noise_slope", 1.0),
            criterion=param_dict.get("criterion", 0.5),
            device=device
        )
        df = model.do_experiment(exp, yt_ids=None, verbose=False)
        all_dfs.append(df)

        if verbose and i % 1 == 0:
            print(f"Finished model simulation for experiment {i}")

    dprime_df = recompute_dprime_by_isi_per_subject(all_dfs, criterion=0)
    model_dprimes = get_dprime_by_isi(dprime_df)  # shape must match human_sensitivity

    if len(model_dprimes) != len(human_sensitivity):
        raise ValueError("Mismatch between model and human d-prime lengths.")

    mse = np.mean((np.array(model_dprimes) - np.array(human_sensitivity)) ** 2)

    if verbose:
        print(f"Model d′: {np.round(model_dprimes, 3)}")
        print(f"Human d′: {np.round(human_sensitivity, 3)}")
        print(f"MSE: {mse:.4f}")

    return mse

import random 
evaluation_log = []
result_log = []
inits = []

from scipy.optimize import basinhopping

def objective(params):
    param_dict = {"noise_slope": params[0], "criterion": params[1], "noise_offset": params[2]}
    
    mse = evaluate_model_dprime_fit(
        memory_model=NoisyAgeMixtureMemoryModel,
        param_dict=param_dict,
        encoding_model=zscore_projector,
        human_sensitivity=human_sensitivity,
        experiment_list=experiment_list[-10:],
        device="cuda",
        verbose=False
    )
    evaluation_log.append((params[0], params[1], mse))
    return mse

# =======================
# deps (keep at top)
# =======================
import os, json, time, random
from typing import Callable, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# =======================
# core helpers
# =======================
class LoggedObjective:
    def __init__(self, objective, run_id, sink):
        self.objective = objective
        self.run_id = run_id
        self.sink = sink
        self.call = 0

    def __call__(self, x):
        mse = self.objective(x)
        # x = [noise_variance, criterion, noise_offset]
        row = {
            "run": self.run_id,
            "call": self.call,
            "noise_variance": float(x[0]),
            "criterion": float(x[1]),
            "noise_offset": float(x[2]),
            "mse": float(mse),
        }
        self.sink.append(row)
        self.call += 1
        return mse


def run_multi_start_minimize(
    objective: Callable[[np.ndarray], float],
    n_inits: int,
    bounds: List[Tuple[float, float]],
    method: str = "Powell",
    out_dir: str = None,
    seed: int = 123,
) -> Dict[str, Any]:
    """
    Run multiple random-initialized optimizations, logging all evaluations.

    Returns:
        {
          "eval_df": DataFrame of ALL (run,call,params,mse),
          "runs_df": DataFrame of per-run {x0, best, mse, success, nfev, nit},
          "best_overall": {"x": [nv,crit], "mse": float, "run": int}
        }
    """
    rng = random.Random(seed)
    eval_log: List[Dict[str, Any]] = []
    runs_out: List[Dict[str, Any]] = []

    for i in range(n_inits):
        x0 = [rng.uniform(bounds[0][0], bounds[0][1]),
              rng.uniform(bounds[1][0], bounds[1][1]), 
             rng.uniform(bounds[2][0], bounds[2][1])]
        logged_obj = LoggedObjective(objective, run_id=i, sink=eval_log)
        res = minimize(
            logged_obj,
            x0=np.array(x0, dtype=float),
            bounds=bounds,
            method=method
        )
        runs_out.append({
            "run": i,
            "x0_noise_variance": float(x0[0]),
            "x0_criterion": float(x0[1]),
            "x0_noise_offset": float(x0[2]),                # NEW
            "best_noise_variance": float(res.x[0]),
            "best_criterion": float(res.x[1]),
            "best_noise_offset": float(res.x[2]),           # NEW
            "best_mse": float(res.fun),
            "success": bool(res.success),
            "status": int(res.status),
            "message": str(res.message),
            "nfev": int(getattr(res, "nfev", -1)),
            "nit": int(getattr(res, "nit", -1)),
        })

    eval_df = pd.DataFrame(eval_log)
    runs_df = pd.DataFrame(runs_out)
    best_idx = int(runs_df["best_mse"].argmin())
    best_overall = {
        "x": [
            float(runs_df.loc[best_idx, "best_noise_variance"]),
            float(runs_df.loc[best_idx, "best_criterion"]),
            float(runs_df.loc[best_idx, "best_noise_offset"]),  # NEW
        ],
        "mse": float(runs_df.loc[best_idx, "best_mse"]),
        "run": int(runs_df.loc[best_idx, "run"])
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        eval_csv = os.path.join(out_dir, f"eval_log_{ts}.csv")
        runs_csv = os.path.join(out_dir, f"runs_summary_{ts}.csv")
        best_json = os.path.join(out_dir, f"best_overall_{ts}.json")
        eval_df.to_csv(eval_csv, index=False)
        runs_df.to_csv(runs_csv, index=False)
        with open(best_json, "w") as f:
            json.dump(best_overall, f, indent=2)
        print(f"[saved] {eval_csv}\n[saved] {runs_csv}\n[saved] {best_json}")

    return {"eval_df": eval_df, "runs_df": runs_df, "best_overall": best_overall}


def marginalize_mse(
    eval_df: pd.DataFrame,
    param_col: str,
    n_bins: int = 30,
    agg: Tuple[str, ...] = ("mean", "median", "min", "count"),
) -> pd.DataFrame:
    """
    Bin one parameter and aggregate MSE across the other, i.e., p(MSE | bin(param)).

    Args:
        eval_df: rows with columns ["noise_variance","criterion","mse", ...]
        param_col: "noise_variance" or "criterion"
        n_bins: number of bins for the chosen param
        agg: aggregations to compute over mse within each bin

    Returns:
        DataFrame with columns [param_col+"_bin_left", param_col+"_bin_right",
                                param_col+"_bin_center", "mse_<agg>"...]
    """
    assert param_col in ("noise_variance", "criterion", "noise_offset")
    x = eval_df[param_col].to_numpy()
    y = eval_df["mse"].to_numpy()

    # Build bins that cover observed range (robust to explorations)
    lo, hi = float(np.min(x)), float(np.max(x))
    if lo == hi:  # degenerate case
        lo, hi = (lo - 1e-6, hi + 1e-6)

    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.clip(np.digitize(x, edges) - 1, 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = (idx == b)
        if not np.any(mask):
            continue
        vals = y[mask]
        row = {
            f"{param_col}_bin_left": float(edges[b]),
            f"{param_col}_bin_right": float(edges[b + 1]),
            f"{param_col}_bin_center": float(0.5 * (edges[b] + edges[b + 1])),
        }
        if "mean" in agg:
            row["mse_mean"] = float(np.mean(vals))
        if "median" in agg:
            row["mse_median"] = float(np.median(vals))
        if "min" in agg:
            row["mse_min"] = float(np.min(vals))
        if "count" in agg:
            row["count"] = int(vals.size)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(f"{param_col}_bin_center").reset_index(drop=True)
    return out


def save_marginals(eval_df: pd.DataFrame, out_dir: str, prefix: str = "marginal", n_bins: int = 30) -> Tuple[str, str, str]:
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")

    nv_df = marginalize_mse(eval_df, "noise_variance", n_bins=n_bins)
    cr_df = marginalize_mse(eval_df, "criterion", n_bins=n_bins)
    no_df = marginalize_mse(eval_df, "noise_offset", n_bins=n_bins)     # NEW

    nv_csv = os.path.join(out_dir, f"{prefix}_noise_variance_{ts}.csv")
    cr_csv = os.path.join(out_dir, f"{prefix}_criterion_{ts}.csv")
    no_csv = os.path.join(out_dir, f"{prefix}_noise_offset_{ts}.csv")   # NEW

    nv_df.to_csv(nv_csv, index=False)
    cr_df.to_csv(cr_csv, index=False)
    no_df.to_csv(no_csv, index=False)                                   # NEW

    print(f"[saved] {nv_csv}\n[saved] {cr_csv}\n[saved] {no_csv}")
    return nv_csv, cr_csv, no_csv


# =======================
# usage in your script
# =======================
# Replace your loop with:
#
# results = run_multi_start_minimize(
#     objective=objective,                      # your existing objective(params)->mse
#     n_inits=20,
#     bounds=[(0.0, 1.0), (0.0, 100.0)],
#     method="Powell",
#     out_dir="/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/figures/human-results/isi-16-only/atexts-len120",
#     seed=123
# )
#
# # Save marginals for quick inspection later
# save_marginals(results["eval_df"], out_dir="/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/figures/human-results/isi-16-only/atexts-len120", n_bins=30)
#
# # Best overall:
# print("BEST:", results["best_overall"])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# grabbing example list of sound
sounds_list = glob.glob("/mindhive/mcdermott/www/mturk_stimuli/bjmedina/mem_exp_atexts_p1/*wav")
texture_list = sounds_list

ALL_SOUNDS = glob.glob("/om2/data/public/audioset/wavs/unbalanced_train_segments_downloads/unbalanced_train_segments_downloads_*/*wav")
print(len(ALL_SOUNDS))

tasks = ["ind-nature-len120" ,"global-music-len120", "atexts-len120", "nhs-region-len120"]
which_task = tasks[2] # "global-music-len120", "atexts-len120" "nhs-region-len120"

base_path = "/mindhive/mcdermott/www/mturk_stimuli/bjmedina/{}/sequences/isi_16/len120/"

seqs_paths = {"ind-nature-len120": "mem_exp_ind-nature_2025", 
              "global-music-len120": "global-music-2025-n_80",
              "atexts-len120": "mem_exp_atexts_2025",
              "nhs-region-len120": "nhs-region-n_80"}

hr_task_name = {"ind-nature-len120": "Industrial and Nature", 
              "global-music-len120": "Globalized Music",
              "atexts-len120": "Auditory Textures",
              "nhs-region-len120": " 'Natural History of Song' "}

exps, seqs, fnames = load_results(f"/mindhive/mcdermott/www/bjmedina/experiments/bolivia_2025/results/isi_16/{which_task}")
move_sequences_to_used(base_path.format(seqs_paths[which_task]), seqs)

texture_model = encoders.AudioTextureEncoder(
    statistics_dict=statistics_dict,
    model_params=model_params,
    sr=20000,
    rms_level=0.05,
    duration=2.0,
    device=device
)

zscore_projector = encoders.ZScoreSpace(texture_model, device=device)
zscore_projector.fit(texture_list) # need to make this a much larger set to get a better estimate of the 
# mean and std

from sklearn.metrics import mean_squared_error

# Your experiment structure (list of stimulus filepaths for each run)
experiment_list = []
for exp in exps:
    list_to_add = []
    for stim in exp.stimulus.tolist():
        edited_stim_name = "/mindhive/mcdermott/www/" + "/".join(stim.split("/")[3:])
        list_to_add.append(edited_stim_name)
    experiment_list.append(list_to_add)



exps, seqs, fnames = load_results_with_exclusion(f"/mindhive/mcdermott/www/bjmedina/experiments/bolivia_2025/results/isi_16/{which_task}",
                                                    min_dprime=2,
                                                    min_trials=120,
                                                    skip_len60=True,
                                                    verbose=False,
                                                    return_skipped=False)



move_sequences_to_used(base_path.format(seqs_paths[which_task]), seqs)

print("Number of participants used in analysis:", len(exps))


safe_name = which_task.lower().replace(" ", "_")  # e.g., "globalized_music"
save_dir = os.path.join("/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/figures/human-results/isi-16-only", safe_name)

ensure_dir(save_dir)
print(save_dir)

human_results = recompute_dprime_by_isi_per_subject(exps)
human_sensitivity = get_dprime_by_isi(human_results)

results = run_multi_start_minimize(
    objective=objective,                      # your existing objective(params)->mse
    n_inits=10,
    bounds=[(0, 1.0), (0, 1.0), (0, 1.0)],
    method="Powell",
    out_dir="/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/figures/human-results/isi-16-only/atexts-len120/noisyage_mixturemodel_relative",
    seed=123
)

# Save marginals for quick inspection later
save_marginals(results["eval_df"], out_dir="/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/figures/human-results/isi-16-only/atexts-len120/noisyage_mixturemodel_relative", n_bins=30)

# Best overall:
print("BEST:", results["best_overall"])
