#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utls.runners — simulation runners for memory experiments
"""

import pandas as pd
import os
import numpy as np
import torch
from collections import defaultdict
from itertools import product

def is_valid_param_combo(params):
    mode = params.get("noise_mode")

    # two-regime noise ignores "rate" → skip any rate that is not needed
    if mode == "two-regime" or mode == "three-regime":
        # you can either:
        return params.get("rate", None) in (None, 0)
        # OR simply:
        # return False if there's a rate, True otherwise
        # return params.get("rate") is None

    # diffuse has no rate parameter → skip if rate is provided
    if mode == "diffuse":
        return params.get("rate") in (None, 0)

    # Add additional rules for other custom schedules...
    return True


def compute_score(probe, mem_mu, std, metric):
    """
    Compute similarity or distance between `probe` and memory mean `mem_mu`
    under the given metric.
    """
    diff = probe - mem_mu
    sqdist = torch.sum(diff**2).item()      # Euclidean squared
    var = std ** 2

    if metric == "mahalanobis":
        return (sqdist ** 0.5) / std

    elif metric == "loglikelihood":
        D = probe.numel()
        return (
            -0.5 * D * np.log(2 * np.pi)
            - D * np.log(std)
            - 0.5 * (sqdist / var)
        )

    elif metric == "euclidean":
        return torch.norm(diff).item()

    elif metric == "manhattan":
        return torch.sum(torch.abs(diff)).item()

    elif metric == "cosine":
        probe_norm = probe / (torch.norm(probe) + 1e-12)
        mem_norm   = mem_mu / (torch.norm(mem_mu) + 1e-12)
        cos_sim    = torch.dot(probe_norm.squeeze(), mem_norm.squeeze()).item()
        return 1.0 - cos_sim  # lower = more similar, like Euclidean

    else:
        raise ValueError(f"Unknown metric '{metric}'")

def make_noise_schedule(noise_mode, params):
    """
    Factory: builds the correct NoiseSchedule object from
    a noise_mode string and the model parameters dictionary.

    Expected params fields:
        - sigma0  (always required)
        - rate    (for power/exp schedules)
        - sigma1  (for two-regime schedules)
        - sigma2  (for three-regime schedules)
        - t_step  (for two-regime schedules)
    """

    sigma0 = params.get("sigma0")
    rate   = params.get("rate")
    sigma1 = params.get("sigma1", None)
    sigma2 = params.get("sigma2", None)
    t_step = params.get("t_step", 5)

    if noise_mode == "constant":
        return ConstantNoise(sigma0)

    elif noise_mode == "diffuse":
        return DiffuseNoise(sigma0)

    elif noise_mode == "power-law":
        return PowerLawNoise(sigma0, rate)

    elif noise_mode == "power-decay":
        return PowerDecayNoise(sigma0, rate)

    elif noise_mode == "two-regime":
        if sigma1 is None:
            raise ValueError("noise_mode='two-regime' requires sigma1")
        return TwoRegimeNoise(sigma0, sigma1, t_step)
        
    elif noise_mode == "three-regime":
        if sigma1 is None:
            raise ValueError("noise_mode='three-regime' requires sigma1 and sigma2")
        return ThreeRegimeNoise(sigma0, sigma1, sigma2, t_step)

    else:
        raise ValueError(f"Unknown noise_mode '{noise_mode}'")

def filter_kwargs_for_core(kwargs):
    """Remove parameters that run_model_core does not accept."""
    allowed = {
        "sigma0", "X0", "name_to_idx", "experiment_list",
        "metric", "noise_schedule",
        "return_item_scores", "return_binary_matrix",
        "decision_threshold", "debug", "seed"
    }
    return {k: v for k, v in kwargs.items() if k in allowed}

class NoiseSchedule:
    """Base class for noise schedules."""
    def __call__(self, age: int) -> float:
        raise NotImplementedError

class DiffuseNoise(NoiseSchedule):
    def __init__(self, sigma0):
        self.sigma0 = sigma0

    def __call__(self, age):
        if age <= 0: return 1e-6
        return max(self.sigma0 * np.sqrt(age), 1e-10)


class PowerLawNoise(NoiseSchedule):
    def __init__(self, sigma0, rate):
        self.sigma0 = sigma0
        self.rate = 0.5 if rate is None else rate

    def __call__(self, age):
        if age <= 0: return 1e-10
        return max(self.sigma0 * (age ** self.rate), 1e-10)


class PowerDecayNoise(NoiseSchedule):
    def __init__(self, sigma0, rate):
        self.sigma0 = sigma0
        self.rate = 0.5 if rate is None else rate

    def __call__(self, age):
        if age <= 0: return 1e-10
        return max(self.sigma0 / (age ** self.rate), 1e-10)

class TwoRegimeNoise(NoiseSchedule):
    """
    Noise increases with sigma0 until t_step, then switches to sigma1.
    """
    def __init__(self, sigma0, sigma1, t_step):
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.t_step = t_step

    def __call__(self, age):
        if age < 0: return 1e-9
        if age < self.t_step:
            return max(self.sigma0, 1e-10)
        return max(self.sigma1, 1e-10)


# class ThreeRegimeNoise(NoiseSchedule):
#     """
#     Drift std is sigma0 for age==1 (first update),
#     sigma1 for 1<age<t_step,
#     sigma2 for age>=t_step.
#     """
#     def __init__(self, sigma0, sigma1, sigma2, t_step):
#         self.sigma0 = sigma0
#         self.sigma1 = sigma1
#         self.sigma2 = sigma2
#         self.t_step = t_step

#     def __call__(self, age):
#         if age < 0: return 1e-9
#         if age == 1: return max(self.sigma0, 1e-10)
#         if age < self.t_step:
#             return max(self.sigma1, 1e-10)
#         return max(self.sigma2, 1e-10)

class ThreeRegimeNoise(NoiseSchedule):
    """
    Drift noise applies only AFTER encoding.

    Encoding noise:
        - applied once at insertion (sigma0)

    Drift noise:
        - sigma1 for 1 < age < t_step
        - sigma2 for age >= t_step
    """
    def __init__(self, sigma0, sigma1, sigma2, t_step):
        self.sigma0 = sigma0
        self.sigma1 = sigma1
        self.sigma2 = sigma2
        self.t_step = t_step

    def __call__(self, age):
        if age <= 1:
            return 1e-5
        if age < self.t_step:
            return max(self.sigma1, 1e-10)
        return max(self.sigma2, 1e-10)

class ConstantNoise(NoiseSchedule):
    """
    Constant noise across all ages.
    """
    def __init__(self, sigma0):
        self.sigma0 = sigma0

    def __call__(self, age):
        if age <= 0:
            return 1e-10
        return max(self.sigma0, 1e-10)


def run_model_core(
    sigma0,
    *,
    X0,
    name_to_idx,
    experiment_list,
    metric="mahalanobis",
    noise_schedule=None,
    return_item_scores=False,
    return_binary_matrix=False,
    decision_threshold=None,
    debug=False,
    torch_rng=None,
    seed=0,
):
    """
    The unified engine for all memory-model simulations.

    Modes:
      - return_item_scores=False, return_binary_matrix=False
            → returns trialwise hit_scores, fa_scores
      - return_item_scores=True, return_binary_matrix=False
            → returns item_hits, item_fas score lists
      - return_binary_matrix=True
            → returns DataFrames with yes/no per item
    """
    # ---- convenience maps ----

    if torch_rng is None:
        torch_rng = torch.Generator(device=X0.device)
        torch_rng.manual_seed(seed)
        
    idx_to_name = {v: k for k, v in name_to_idx.items()}
    D = X0.shape[1]

    # # heteroskedastic scaling
    # dim_std = X0.std(0, unbiased=True)
    # scaled_std = dim_std / dim_std.max()

    dim_std = X0.std(0, unbiased=True)

    # RMS-normalized feature scaling 
    rms_std = torch.sqrt(torch.mean(dim_std ** 2))
    scaled_std = dim_std / rms_std
    

    # outputs
    hit_scores, fa_scores = [], []
    isi_hit_dists = defaultdict(list)
    T_max = max((len(seq) for seq in experiment_list), default=0)
    fa_by_t = [[] for _ in range(T_max)]
    
    item_hits, item_fas = defaultdict(list), defaultdict(list)
    binary_hits, binary_fas = [], []

    stds_over_time = []

    # for binary mode: list of filenames for columns
    all_fnames = sorted(name_to_idx.keys())

    for seq in experiment_list:
        if not seq:
            continue

        seq_idx = [name_to_idx[f] for f in seq]
        memory_bank, seen, last_seen = [], set(), {}

        # For binary mode, init one row per sequence
        if return_binary_matrix:
            row_hits = {os.path.basename(f): np.nan for f in all_fnames}
            row_fas  = {os.path.basename(f): np.nan for f in all_fnames}

        for t, incoming in enumerate(seq_idx, start=1):
            probe = X0[incoming].view(1, -1)
            fname = idx_to_name[incoming]
            scores = []

            # ------------------ UPDATE MEMORIES ------------------
            for mem in memory_bank:
                age = t - mem["t_inserted"]
                # if age <= 0: 
                #     continue

                std = noise_schedule(age)
                stds_over_time.append((age, std))

                # drift
                noise = torch.randn(
                    mem["mu"].shape,
                    device=mem["mu"].device,
                    dtype=mem["mu"].dtype,
                    generator=torch_rng,
                ) * (std * scaled_std)
                
                mem["mu"] += noise

                # score
                score = compute_score(probe, mem["mu"], std, metric)
                scores.append(score)


            # ------------------ DECISION STEP ------------------
            if scores:
                score_val = max(scores) if metric == "loglikelihood" else min(scores)
                if debug:
                    print(f"[ BEST SCORE ]: {score}")
                is_repeat = incoming in seen

                # TRIALWISE MODE
                if not return_item_scores and not return_binary_matrix:
                    if is_repeat:
                        hit_scores.append(score_val)
                        isi = t - last_seen[incoming]
                        isi_hit_dists[isi].append((score_val, t))
                    else:
                        fa_scores.append(score_val)
                        fa_by_t[t - 1].append(score_val)

                # ITEMWISE SCORE MODE
                if return_item_scores and not return_binary_matrix:

                    if is_repeat:
                        isi = t - last_seen[incoming]
                        if isi > 1:
                            item_hits[fname].append(score_val)
                    else:
                        item_fas[fname].append(score_val)

                # BINARY MATRIX MODE
                if return_binary_matrix:
                    if decision_threshold is not None:
                        yes = 1 if score_val <= decision_threshold else 0
                    else:
                        yes = 1 if is_repeat else 0

                    if is_repeat:
                        row_hits[os.path.basename(fname)] = yes
                    else:
                        row_fas[os.path.basename(fname)] = yes

            # ------------------ STORE NEW MEMORY ------------------
            base = X0[incoming].clone()
            noise = torch.randn(
                base.shape,
                device=base.device,
                dtype=base.dtype,
                generator=torch_rng,
            )
            if debug:
                print( "noise, sigma0" ) 
                print( torch.sum(noise), sigma0 )
            mem = base + noise * (sigma0 * dim_std)
            memory_bank.append({"mu": mem.view(1, -1), "t_inserted": t})
            seen.add(incoming)
            last_seen[incoming] = t

        if return_binary_matrix:
            binary_hits.append(row_hits)
            binary_fas.append(row_fas)

    # ------------------ RETURN ------------------
    out = {
        "stds_over_time": np.array(stds_over_time),
        "metric": metric,
        "isi_hit_dists": isi_hit_dists,
        "score_type": "likelihood" if metric == "loglikelihood" else "distance",
        "fa_by_t": fa_by_t,
        "T_max": T_max,
    }

    if not return_binary_matrix:
        out.update({
            "hits": np.array(hit_scores),
            "fas": np.array(fa_scores),
            "item_hits": item_hits,
            "item_fas": item_fas,
        })

    if return_binary_matrix:
        out.update({
            "hits": pd.DataFrame(binary_hits),
            "fas": pd.DataFrame(binary_fas),
        })

    return out

def run_experiment_scores( debug=False, seed=0, **kwargs,):
    schedule = make_noise_schedule(kwargs["noise_mode"], kwargs)
    core_kwargs = filter_kwargs_for_core(kwargs)
    return run_model_core(
        **core_kwargs,
        noise_schedule=schedule,
        return_item_scores=False,
        return_binary_matrix=False,
        debug=debug, 
        seed=seed
    )

def run_experiment_scores_v2( debug=False, seed=0, **kwargs,):
    schedule = make_noise_schedule(kwargs["noise_mode"], kwargs)
    core_kwargs = filter_kwargs_for_core(kwargs)
    return run_model_core_v2(
        **core_kwargs,
        noise_schedule=schedule,
        return_item_scores=False,
        return_binary_matrix=False,
        debug=debug, 
        seed=seed
    )

def run_experiment_scores_v3( debug=False, seed=0, **kwargs,):
    schedule = make_noise_schedule(kwargs["noise_mode"], kwargs)
    core_kwargs = filter_kwargs_for_core(kwargs)
    return run_model_core_v3(
        **core_kwargs,
        noise_schedule=schedule,
        return_item_scores=False,
        return_binary_matrix=False,
        debug=debug, 
        seed=seed
    )

    run_model_core_v4

def run_experiment_scores_v4( debug=False, seed=0, **kwargs,):
    schedule = make_noise_schedule(kwargs["noise_mode"], kwargs)
    core_kwargs = filter_kwargs_for_core(kwargs)
    return run_model_core_v4(
        **core_kwargs,
        noise_schedule=schedule,
        return_item_scores=False,
        return_binary_matrix=False,
        debug=debug, 
        seed=seed
    )

def run_experiment_scores_itemwise(**kwargs):
    schedule = make_noise_schedule(kwargs["noise_mode"], kwargs)
    core_kwargs = filter_kwargs_for_core(kwargs)
    return run_model_core(
        **core_kwargs,
        noise_schedule=schedule,
        return_item_scores=True,
        return_binary_matrix=False
    )

def run_experiment_scores_itemwise_v2(**kwargs):
    schedule = make_noise_schedule(kwargs["noise_mode"], kwargs)
    core_kwargs = filter_kwargs_for_core(kwargs)
    return run_model_core_v2(
        **core_kwargs,
        noise_schedule=schedule,
        return_item_scores=True,
        return_binary_matrix=False
    )

def run_experiment_itemwise_hits_fas(**kwargs):
    schedule = make_noise_schedule(kwargs["noise_mode"], kwargs)
    core_kwargs = filter_kwargs_for_core(kwargs)
    return run_model_core(
        **core_kwargs,
        noise_schedule=schedule,
        return_item_scores=False,
        return_binary_matrix=True,
        decision_threshold=kwargs.get("decision_threshold", None)
    )

def run_model_core_v2(
    sigma0,
    *,
    X0,
    name_to_idx,
    experiment_list,
    metric="mahalanobis",
    noise_schedule=None,
    return_item_scores=False,
    return_binary_matrix=False,
    decision_threshold=None,
    debug=False,
    torch_rng=None,
    seed=0,
):
    """
    The unified engine for all memory-model simulations.

    Modes:
      - return_item_scores=False, return_binary_matrix=False
            → returns trialwise hit_scores, fa_scores
      - return_item_scores=True, return_binary_matrix=False
            → returns item_hits, item_fas score lists
      - return_binary_matrix=True
            → returns DataFrames with yes/no per item
    """
    # ---- convenience maps ----

    if torch_rng is None:
        torch_rng = torch.Generator(device=X0.device)
        torch_rng.manual_seed(seed)
        
    idx_to_name = {v: k for k, v in name_to_idx.items()}
    D = X0.shape[1]

    # # heteroskedastic scaling
    # dim_std = X0.std(0, unbiased=True)
    # scaled_std = dim_std / dim_std.max()

    dim_std = X0.std(0, unbiased=True)

    # RMS-normalized feature scaling 
    rms_std = torch.sqrt(torch.mean(dim_std ** 2))
    scaled_std = dim_std / rms_std
    

    # outputs
    hit_scores, fa_scores = [], []
    isi_hit_dists = defaultdict(list)
    T_max = max((len(seq) for seq in experiment_list), default=0)
    fa_by_t = [[] for _ in range(T_max)]
    
    item_hits, item_fas = defaultdict(list), defaultdict(list)
    binary_hits, binary_fas = [], []

    stds_over_time = []

    # for binary mode: list of filenames for columns
    all_fnames = sorted(name_to_idx.keys())

    for seq in experiment_list:
        if not seq:
            continue

        seq_idx = [name_to_idx[f] for f in seq]
        memory_bank, seen, last_seen = [], set(), {}

        # For binary mode, init one row per sequence
        if return_binary_matrix:
            row_hits = {os.path.basename(f): np.nan for f in all_fnames}
            row_fas  = {os.path.basename(f): np.nan for f in all_fnames}

        for t, incoming in enumerate(seq_idx, start=1):
            probe = X0[incoming].view(1, -1)
            fname = idx_to_name[incoming]
            scores = []

            # ------------------ UPDATE MEMORIES ------------------
            for mem in memory_bank:
                age = t - mem["t_inserted"]
                # if age <= 0: 
                #     continue

                std = noise_schedule(age)
                stds_over_time.append((age, std))

                # drift
                noise = torch.randn(
                    mem["mu"].shape,
                    device=mem["mu"].device,
                    dtype=mem["mu"].dtype,
                    generator=torch_rng,
                ) * (std * scaled_std)
                
                mem["mu"] += noise

                # score
                score = compute_score(probe, mem["mu"], std, metric)
                scores.append(score)


            # ------------------ DECISION STEP ------------------
            if scores:
                score_val = max(scores) if metric == "loglikelihood" else np.mean(scores)
                if debug:
                    print(f"[ BEST SCORE ]: {score}")
                is_repeat = incoming in seen

                # TRIALWISE MODE
                if not return_item_scores and not return_binary_matrix:
                    if is_repeat:
                        hit_scores.append(score_val)
                        isi = t - last_seen[incoming]
                        isi_hit_dists[isi].append((score_val, t))
                    else:
                        fa_scores.append(score_val)
                        fa_by_t[t - 1].append(score_val)

                # ITEMWISE SCORE MODE
                if return_item_scores and not return_binary_matrix:

                    if is_repeat:
                        isi = t - last_seen[incoming]
                        if isi > 1:
                            item_hits[fname].append(score_val)
                    else:
                        item_fas[fname].append(score_val)

                # BINARY MATRIX MODE
                if return_binary_matrix:
                    if decision_threshold is not None:
                        yes = 1 if score_val <= decision_threshold else 0
                    else:
                        yes = 1 if is_repeat else 0

                    if is_repeat:
                        row_hits[os.path.basename(fname)] = yes
                    else:
                        row_fas[os.path.basename(fname)] = yes

            # ------------------ STORE NEW MEMORY ------------------
            base = X0[incoming].clone()
            noise = torch.randn(
                base.shape,
                device=base.device,
                dtype=base.dtype,
                generator=torch_rng,
            )
            if debug:
                print( "noise, sigma0" ) 
                print( torch.sum(noise), sigma0 )
            mem = base + noise * (sigma0 * dim_std)
            memory_bank.append({"mu": mem.view(1, -1), "t_inserted": t})
            seen.add(incoming)
            last_seen[incoming] = t

        if return_binary_matrix:
            binary_hits.append(row_hits)
            binary_fas.append(row_fas)

    # ------------------ RETURN ------------------
    out = {
        "stds_over_time": np.array(stds_over_time),
        "metric": metric,
        "isi_hit_dists": isi_hit_dists,
        "score_type": "likelihood" if metric == "loglikelihood" else "distance",
        "fa_by_t": fa_by_t,
        "T_max": T_max,
    }

    if not return_binary_matrix:
        out.update({
            "hits": np.array(hit_scores),
            "fas": np.array(fa_scores),
            "item_hits": item_hits,
            "item_fas": item_fas,
        })

    if return_binary_matrix:
        out.update({
            "hits": pd.DataFrame(binary_hits),
            "fas": pd.DataFrame(binary_fas),
        })

    return out

def run_model_core_v3(
    sigma0,
    *,
    X0,
    name_to_idx,
    experiment_list,
    metric="mahalanobis",
    noise_schedule=None,
    return_item_scores=False,
    return_binary_matrix=False,
    decision_threshold=None,
    debug=False,
    torch_rng=None,
    seed=0,
):
    """
    The unified engine for all memory-model simulations.

    Modes:
      - return_item_scores=False, return_binary_matrix=False
            → returns trialwise hit_scores, fa_scores
      - return_item_scores=True, return_binary_matrix=False
            → returns item_hits, item_fas score lists
      - return_binary_matrix=True
            → returns DataFrames with yes/no per item
    """

    def k_nearest_mean(scores, k):
        """
        Mean of the K smallest scores.
        """
        scores = np.asarray(scores)
        k = min(k, len(scores))
        return np.mean(np.partition(scores, k - 1)[:k])
    # ---- convenience maps ----

    if torch_rng is None:
        torch_rng = torch.Generator(device=X0.device)
        torch_rng.manual_seed(seed)
        
    idx_to_name = {v: k for k, v in name_to_idx.items()}
    D = X0.shape[1]

    # # heteroskedastic scaling
    # dim_std = X0.std(0, unbiased=True)
    # scaled_std = dim_std / dim_std.max()

    dim_std = X0.std(0, unbiased=True)

    # RMS-normalized feature scaling 
    rms_std = torch.sqrt(torch.mean(dim_std ** 2))
    scaled_std = dim_std / rms_std
    

    # outputs
    hit_scores, fa_scores = [], []
    isi_hit_dists = defaultdict(list)
    T_max = max((len(seq) for seq in experiment_list), default=0)
    fa_by_t = [[] for _ in range(T_max)]
    
    item_hits, item_fas = defaultdict(list), defaultdict(list)
    binary_hits, binary_fas = [], []

    stds_over_time = []

    # for binary mode: list of filenames for columns
    all_fnames = sorted(name_to_idx.keys())

    for seq in experiment_list:
        if not seq:
            continue

        seq_idx = [name_to_idx[f] for f in seq]
        memory_bank, seen, last_seen = [], set(), {}

        # For binary mode, init one row per sequence
        if return_binary_matrix:
            row_hits = {os.path.basename(f): np.nan for f in all_fnames}
            row_fas  = {os.path.basename(f): np.nan for f in all_fnames}

        for t, incoming in enumerate(seq_idx, start=1):
            probe = X0[incoming].view(1, -1)
            fname = idx_to_name[incoming]
            scores = []

            # ------------------ UPDATE MEMORIES ------------------
            for mem in memory_bank:
                age = t - mem["t_inserted"]
                # if age <= 0: 
                #     continue

                std = noise_schedule(age)
                stds_over_time.append((age, std))

                # drift
                noise = torch.randn(
                    mem["mu"].shape,
                    device=mem["mu"].device,
                    dtype=mem["mu"].dtype,
                    generator=torch_rng,
                ) * (std * scaled_std)
                
                mem["mu"] += noise

                # score
                score = compute_score(probe, mem["mu"], std, metric)
                scores.append(score)


            # ------------------ DECISION STEP ------------------
            if scores:
                score_val = max(scores) if metric == "loglikelihood" else k_nearest_mean(scores, k=5)
                if debug:
                    print(f"[ BEST SCORE ]: {score}")
                is_repeat = incoming in seen

                # TRIALWISE MODE
                if not return_item_scores and not return_binary_matrix:
                    if is_repeat:
                        hit_scores.append(score_val)
                        isi = t - last_seen[incoming]
                        isi_hit_dists[isi].append((score_val, t))
                    else:
                        fa_scores.append(score_val)
                        fa_by_t[t - 1].append(score_val)

                # ITEMWISE SCORE MODE
                if return_item_scores and not return_binary_matrix:

                    if is_repeat:
                        isi = t - last_seen[incoming]
                        if isi > 1:
                            item_hits[fname].append(score_val)
                    else:
                        item_fas[fname].append(score_val)

                # BINARY MATRIX MODE
                if return_binary_matrix:
                    if decision_threshold is not None:
                        yes = 1 if score_val <= decision_threshold else 0
                    else:
                        yes = 1 if is_repeat else 0

                    if is_repeat:
                        row_hits[os.path.basename(fname)] = yes
                    else:
                        row_fas[os.path.basename(fname)] = yes

            # ------------------ STORE NEW MEMORY ------------------
            base = X0[incoming].clone()
            noise = torch.randn(
                base.shape,
                device=base.device,
                dtype=base.dtype,
                generator=torch_rng,
            )
            if debug:
                print( "noise, sigma0" ) 
                print( torch.sum(noise), sigma0 )
            mem = base + noise * (sigma0 * dim_std)
            memory_bank.append({"mu": mem.view(1, -1), "t_inserted": t})
            seen.add(incoming)
            last_seen[incoming] = t

        if return_binary_matrix:
            binary_hits.append(row_hits)
            binary_fas.append(row_fas)

    # ------------------ RETURN ------------------
    out = {
        "stds_over_time": np.array(stds_over_time),
        "metric": metric,
        "isi_hit_dists": isi_hit_dists,
        "score_type": "likelihood" if metric == "loglikelihood" else "distance",
        "fa_by_t": fa_by_t,
        "T_max": T_max,
    }

    if not return_binary_matrix:
        out.update({
            "hits": np.array(hit_scores),
            "fas": np.array(fa_scores),
            "item_hits": item_hits,
            "item_fas": item_fas,
        })

    if return_binary_matrix:
        out.update({
            "hits": pd.DataFrame(binary_hits),
            "fas": pd.DataFrame(binary_fas),
        })

    return out

def run_model_core_v4(
    sigma0,
    *,
    X0,
    name_to_idx,
    experiment_list,
    metric="mahalanobis",
    noise_schedule=None,
    return_item_scores=False,
    return_binary_matrix=False,
    decision_threshold=None,
    debug=False,
    torch_rng=None,
    seed=0,
):
    """
    The unified engine for memory-model simulations.
    Decision score = score to most recent memory trace.
    """

    if torch_rng is None:
        torch_rng = torch.Generator(device=X0.device)
        torch_rng.manual_seed(seed)

    idx_to_name = {v: k for k, v in name_to_idx.items()}

    dim_std = X0.std(0, unbiased=True)
    rms_std = torch.sqrt(torch.mean(dim_std ** 2))
    scaled_std = dim_std / rms_std

    hit_scores, fa_scores = [], []
    isi_hit_dists = defaultdict(list)
    T_max = max((len(seq) for seq in experiment_list), default=0)
    fa_by_t = [[] for _ in range(T_max)]

    item_hits, item_fas = defaultdict(list), defaultdict(list)
    binary_hits, binary_fas = [], []

    stds_over_time = []
    all_fnames = sorted(name_to_idx.keys())

    for seq in experiment_list:
        if not seq:
            continue

        seq_idx = [name_to_idx[f] for f in seq]
        memory_bank, seen, last_seen = [], set(), {}

        if return_binary_matrix:
            row_hits = {os.path.basename(f): np.nan for f in all_fnames}
            row_fas  = {os.path.basename(f): np.nan for f in all_fnames}

        for t, incoming in enumerate(seq_idx, start=1):
            probe = X0[incoming].view(1, -1)
            fname = idx_to_name[incoming]

            # ------------------ UPDATE MEMORIES ------------------
            for mem in memory_bank:
                age = t - mem["t_inserted"]
                std = noise_schedule(age)
                stds_over_time.append((age, std))

                noise = torch.randn(
                    mem["mu"].shape,
                    device=mem["mu"].device,
                    dtype=mem["mu"].dtype,
                    generator=torch_rng,
                ) * (std * scaled_std)

                mem["mu"] += noise

            # ------------------ DECISION STEP ------------------
            if memory_bank:
                last_mem = memory_bank[-1]
                age = t - last_mem["t_inserted"]
                std = noise_schedule(age)

                score_val = compute_score(
                    probe,
                    last_mem["mu"],
                    std,
                    metric,
                )

                is_repeat = incoming in seen

                if not return_item_scores and not return_binary_matrix:
                    if is_repeat:
                        hit_scores.append(score_val)
                        isi = t - last_seen[incoming]
                        isi_hit_dists[isi].append((score_val, t))
                    else:
                        fa_scores.append(score_val)
                        fa_by_t[t - 1].append(score_val)

                if return_item_scores and not return_binary_matrix:
                    if is_repeat:
                        isi = t - last_seen[incoming]
                        if isi > 1:
                            item_hits[fname].append(score_val)
                    else:
                        item_fas[fname].append(score_val)

                if return_binary_matrix:
                    if decision_threshold is not None:
                        yes = 1 if score_val <= decision_threshold else 0
                    else:
                        yes = 1 if is_repeat else 0

                    if is_repeat:
                        row_hits[os.path.basename(fname)] = yes
                    else:
                        row_fas[os.path.basename(fname)] = yes

            # ------------------ STORE NEW MEMORY ------------------
            base = X0[incoming].clone()
            noise = torch.randn(
                base.shape,
                device=base.device,
                dtype=base.dtype,
                generator=torch_rng,
            )

            mem = base + noise * (sigma0 * dim_std)
            memory_bank.append({"mu": mem.view(1, -1), "t_inserted": t})
            seen.add(incoming)
            last_seen[incoming] = t

        if return_binary_matrix:
            binary_hits.append(row_hits)
            binary_fas.append(row_fas)

    out = {
        "stds_over_time": np.array(stds_over_time),
        "metric": metric,
        "isi_hit_dists": isi_hit_dists,
        "score_type": "likelihood" if metric == "loglikelihood" else "distance",
        "fa_by_t": fa_by_t,
        "T_max": T_max,
    }

    if not return_binary_matrix:
        out.update({
            "hits": np.array(hit_scores),
            "fas": np.array(fa_scores),
            "item_hits": item_hits,
            "item_fas": item_fas,
        })

    if return_binary_matrix:
        out.update({
            "hits": pd.DataFrame(binary_hits),
            "fas": pd.DataFrame(binary_fas),
        })

    return out

def run_experiment_grid(
    model_fn,
    *,
    X0,
    name_to_idx,
    experiment_list,
    param_grid,
    fixed_params=None,
    debug=False
):
    """
    Runs a full Cartesian grid search over param_grid.

    param_grid: dict like
        {
            "sigma0": [0.1, 0.2],
            "rate": [0.25, 0.5],
            "noise_mode": ["diffuse", "power-law"],
            "metric": ["cosine"],
        }

    model_fn: function like run_experiment_scores, run_experiment_scores_itemwise, ...
    """
    fixed_params = fixed_params or {}
    results_all = []

    # 1) Get keys and list of value lists
    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    # 2) Cartesian product over all value combinations
    for combo in product(*values):
        combo_params = dict(zip(keys, combo))
        params = {**combo_params, **fixed_params}
    
        # if not is_valid_param_combo(params):
        #     if debug:
        #         print(f"[GRID] Skipping invalid params: {params}")
        #     continue

        if debug:
            print(f"\n[GRID] Running model with params: {params}")

        # 3) Call model function
        out = model_fn(
            X0=X0,
            name_to_idx=name_to_idx,
            experiment_list=experiment_list,
            **params
        )

        # 4) Store results
        results_all.append({
            "params": params,
            "results": out,
        })

    return results_all