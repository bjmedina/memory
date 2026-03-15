"""
utls.runners_2d — 2D guided-drift simulation engine for the mechanistic sandbox.

Implements the two-parameter noise model from the paper:
  - σ₀ : encoding noise (applied once at memory insertion)
  - σ  : diffusive noise (constant per-step noise during Langevin dynamics)

Uses ``make_high_diversity_sequences`` for properly interleaved
experiment sequences (~50% repetition rate, mixed ISIs).
"""

import numpy as np
import torch
from collections import defaultdict

from utls.runners_v2 import compute_score
from utls.toy_experiments import make_high_diversity_sequences
from utls.roc_utils import roc_from_arrays
from utls.analysis_helpers import auroc_to_dprime, bootstrap_dprime_ci


# ── core simulation engine ────────────────────────────────────────────

def run_model_core_2d(
    sigma0,
    sigma,
    *,
    X0,
    name_to_idx,
    experiment_list,
    score_model,
    drift_step_size=0.0,
    metric="cosine",
    debug=False,
    torch_rng=None,
    seed=0,
):
    """2D guided-drift memory simulation.

    Parameters
    ----------
    sigma0 : float
        Encoding noise magnitude (applied once at memory insertion).
    sigma : float
        Diffusive noise magnitude (constant per step during Langevin dynamics).
    X0 : Tensor [N, 2]
        Stimulus embeddings.
    name_to_idx : dict
        Stimulus name → row index in *X0*.
    experiment_list : list[list[str]]
        Sequences of stimulus names.
    score_model : ScoreAdapter2D
        Provides ``.forward(x)`` and ``.forward_raw(x)``.
    drift_step_size : float
        Magnitude of prior-driven drift per trial (η in the paper).
    metric : str
        Distance metric for decision (default ``"cosine"``).
    debug : bool
    torch_rng : torch.Generator or None
    seed : int

    Returns
    -------
    dict
        Keys: ``hits``, ``fas``, ``isi_hit_dists``, ``fa_by_t``,
        ``T_max``, ``score_type``, ``stds_over_time``, ``metric``.
    """
    if torch_rng is None:
        torch_rng = torch.Generator(device=X0.device)
        torch_rng.manual_seed(seed)

    idx_to_name = {v: k for k, v in name_to_idx.items()}
    D = X0.shape[1]

    dim_std = X0.std(0, unbiased=True)
    rms_std = torch.sqrt(torch.mean(dim_std ** 2)).item()
    scaled_std = dim_std / rms_std

    hit_scores, fa_scores = [], []
    isi_hit_dists = defaultdict(list)
    T_max = max((len(seq) for seq in experiment_list), default=0)
    fa_by_t = [[] for _ in range(T_max)]
    stds_over_time = []

    for seq in experiment_list:
        if not seq:
            continue

        seq_idx = [name_to_idx[f] for f in seq]
        memory_bank, seen, last_seen = [], set(), {}

        for t, incoming in enumerate(seq_idx, start=1):
            probe = X0[incoming].view(1, -1)
            scores = []

            # ---------- UPDATE MEMORIES ----------
            if memory_bank:
                # random noise (batched) — per-step σ
                mu_dtype = memory_bank[0]["mu"].dtype
                mu_device = memory_bank[0]["mu"].device
                noise_batch = torch.randn(
                    len(memory_bank), D,
                    device=mu_device, dtype=mu_dtype,
                    generator=torch_rng,
                )
                for i in range(len(memory_bank)):
                    memory_bank[i]["mu"] += noise_batch[i:i + 1] * (sigma * scaled_std)

                # prior-driven drift (batched)
                if drift_step_size > 0:
                    mu_batch = torch.cat(
                        [mem["mu"] for mem in memory_bank], dim=0
                    )
                    with torch.no_grad():
                        drift_batch = score_model.forward(mu_batch)
                    for i, mem in enumerate(memory_bank):
                        mem["mu"] += drift_step_size * drift_batch[i].view_as(
                            mem["mu"]
                        )

                # decision scores
                for mem in memory_bank:
                    score = compute_score(probe, mem["mu"], sigma, metric)
                    scores.append(score)

            # ---------- DECISION STEP ----------
            if scores:
                score_val = (
                    max(scores) if metric == "loglikelihood" else min(scores)
                )
                is_repeat = incoming in seen

                if is_repeat:
                    hit_scores.append(score_val)
                    isi = t - last_seen[incoming]
                    isi_hit_dists[isi].append((score_val, t))
                else:
                    fa_scores.append(score_val)
                    fa_by_t[t - 1].append(score_val)

            # ---------- STORE NEW MEMORY ----------
            base = X0[incoming].clone()
            noise = torch.randn(
                base.shape, device=base.device, dtype=base.dtype,
                generator=torch_rng,
            )
            mem_trace = base + noise * (sigma0 * dim_std)
            memory_bank.append({"mu": mem_trace.view(1, -1), "t_inserted": t})
            seen.add(incoming)
            last_seen[incoming] = t

    return {
        "hits": np.array(hit_scores),
        "fas": np.array(fa_scores),
        "isi_hit_dists": dict(isi_hit_dists),
        "fa_by_t": fa_by_t,
        "T_max": T_max,
        "stds_over_time": np.array(stds_over_time) if stds_over_time else np.empty((0, 2)),
        "metric": metric,
        "score_type": "likelihood" if metric == "loglikelihood" else "distance",
    }


# ── high-level ISI sweep ─────────────────────────────────────────────

def run_2d_isi_sweep(
    sigma0,
    sigma,
    drift_step_size,
    score_model,
    X0,
    name_to_idx,
    stimulus_pool,
    isi_values=(0, 1, 2, 4, 8, 16, 32, 64),
    metric="cosine",
    n_sequences=10,
    seq_length=69,
    min_pairs_per_isi=3,
    n_mc=16,
    seed=42,
):
    """Run a full ISI sweep and return d' (mean ± SEM) per ISI.

    Uses ``make_high_diversity_sequences`` for properly interleaved
    experiment sequences with mixed ISIs and ~50% repetition rate.

    Parameters
    ----------
    sigma0 : float
        Encoding noise (applied once at memory insertion).
    sigma : float
        Diffusive noise (constant per-step noise during Langevin dynamics).
    drift_step_size : float
        Prior-driven drift magnitude (η in the paper).
    score_model : ScoreAdapter2D
    X0, name_to_idx, stimulus_pool : as from ``make_2d_grid_stimuli``.
    isi_values : tuple of int
        ISI conditions to evaluate.
    metric : str
        Distance metric for decision (default ``"cosine"``).
    n_sequences : int
        Number of interleaved experiment sequences to generate.
    seq_length : int
        Length of each sequence (must be divisible by 3).
    min_pairs_per_isi : int
        Minimum repeat pairs per ISI condition per sequence.
    n_mc : int
        Monte-Carlo repetitions.
    seed : int

    Returns
    -------
    dict
        ``isi_values`` : list[int]
        ``dprime_mean`` : list[float]
        ``dprime_sem``  : list[float]
        ``auroc``       : list[float]
        ``raw_runs``    : dict[int, dict]  (per-ISI aggregated run data)
    """
    # Generate interleaved multi-ISI sequences
    exp_list, isi_keys = make_high_diversity_sequences(
        stimulus_pool=stimulus_pool,
        isi_values=list(isi_values),
        n_sequences=n_sequences,
        length=seq_length,
        min_pairs_per_isi=min_pairs_per_isi,
        seed=seed,
    )

    score_type = "likelihood" if metric == "loglikelihood" else "distance"

    # Run MC reps and aggregate hits/FAs per ISI
    # Runner ISI = experiment ISI + 1 (offset from t - last_seen)
    runner_isi_values = [isi + 1 for isi in isi_values]

    all_isi_hits = defaultdict(list)  # runner_isi → list of scores
    all_fas = []

    for rep in range(n_mc):
        run_out = run_model_core_2d(
            sigma0=sigma0,
            sigma=sigma,
            X0=X0,
            name_to_idx=name_to_idx,
            experiment_list=exp_list,
            score_model=score_model,
            drift_step_size=drift_step_size,
            metric=metric,
            seed=seed * 10_000 + rep,
        )
        all_fas.extend(run_out["fas"])
        for runner_isi, entries in run_out["isi_hit_dists"].items():
            all_isi_hits[runner_isi].extend([s for s, _ in entries])

    fas_arr = np.asarray(all_fas, float)

    # Compute d' per ISI
    results_isi = []
    results_dprime_mean = []
    results_dprime_sem = []
    results_auroc = []
    raw_runs = {}

    for exp_isi, runner_isi in zip(isi_values, runner_isi_values):
        hits_for_isi = all_isi_hits.get(runner_isi, [])
        if len(hits_for_isi) < 3:
            results_isi.append(exp_isi)
            results_dprime_mean.append(np.nan)
            results_dprime_sem.append(np.nan)
            results_auroc.append(np.nan)
            raw_runs[exp_isi] = {}
            continue

        hits_arr = np.asarray(hits_for_isi, float)

        roc_res = roc_from_arrays(hits_arr, fas_arr, score_type=score_type)
        if roc_res is not None:
            _, _, auc_val = roc_res
            dp = auroc_to_dprime(auc_val)
        else:
            auc_val = np.nan
            dp = np.nan

        # Build run_data for bootstrap
        run_data = {
            "isi_hit_dists": {runner_isi: [(h, 0) for h in hits_arr]},
            "fas": fas_arr,
            "score_type": score_type,
        }
        mean_dp, sem_dp = bootstrap_dprime_ci(run_data, runner_isi, n_boot=200)

        results_isi.append(exp_isi)
        results_dprime_mean.append(float(mean_dp) if np.isfinite(mean_dp) else dp)
        results_dprime_sem.append(float(sem_dp) if np.isfinite(sem_dp) else 0.0)
        results_auroc.append(float(auc_val))
        raw_runs[exp_isi] = run_data

    return {
        "isi_values": results_isi,
        "dprime_mean": results_dprime_mean,
        "dprime_sem": results_dprime_sem,
        "auroc": results_auroc,
        "raw_runs": raw_runs,
    }
