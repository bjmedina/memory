"""
utls.runners_2d — 2D guided-drift simulation engine for the mechanistic sandbox.

Direct mirror of ``run_model_core_prior`` from ``utls/runners_v2.py``,
simplified to the 2D analytic-prior setting (no item-score / binary modes).
"""

import numpy as np
import torch
from collections import defaultdict

from utls.runners_v2 import compute_score, ThreeRegimeNoise
from utls.toy_experiments import make_toy_experiment_list
from utls.roc_utils import roc_from_arrays
from utls.analysis_helpers import auroc_to_dprime, bootstrap_dprime_ci


# ── core simulation engine ────────────────────────────────────────────

def run_model_core_2d(
    sigma0,
    *,
    X0,
    name_to_idx,
    experiment_list,
    score_model,
    drift_step_size=0.0,
    metric="mahalanobis",
    noise_schedule=None,
    debug=False,
    torch_rng=None,
    seed=0,
):
    """2D guided-drift memory simulation.

    Mirrors ``run_model_core_prior`` from ``utls/runners_v2.py``.

    Parameters
    ----------
    sigma0 : float
        Encoding noise magnitude (applied once at memory insertion).
    X0 : Tensor [N, 2]
        Stimulus embeddings.
    name_to_idx : dict
        Stimulus name → row index in *X0*.
    experiment_list : list[list[str]]
        Sequences of stimulus names.
    score_model : ScoreAdapter2D
        Provides ``.forward(x)`` and ``.forward_raw(x)``.
    drift_step_size : float
        Magnitude of prior-driven drift per trial.
    metric : str
        Distance metric for decision (default ``"mahalanobis"``).
    noise_schedule : callable
        Maps age (int) → noise std for that step.
    debug : bool
    torch_rng : torch.Generator or None
    seed : int

    Returns
    -------
    dict
        Same keys as ``run_model_core_prior``: ``hits``, ``fas``,
        ``isi_hit_dists``, ``fa_by_t``, ``T_max``, ``score_type``,
        ``stds_over_time``, ``metric``.
    """
    if torch_rng is None:
        torch_rng = torch.Generator(device=X0.device)
        torch_rng.manual_seed(seed)

    idx_to_name = {v: k for k, v in name_to_idx.items()}
    D = X0.shape[1]

    dim_std = X0.std(0, unbiased=True)
    rms_std = torch.sqrt(torch.mean(dim_std ** 2))
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
            fname = idx_to_name[incoming]
            scores = []

            # ---------- UPDATE MEMORIES ----------
            if memory_bank:
                ages = [t - mem["t_inserted"] for mem in memory_bank]
                stds = [noise_schedule(age) for age in ages]

                for age, std in zip(ages, stds):
                    stds_over_time.append((age, std))

                # random noise (batched)
                mu_dtype = memory_bank[0]["mu"].dtype
                mu_device = memory_bank[0]["mu"].device
                noise_batch = torch.randn(
                    len(memory_bank), D,
                    device=mu_device, dtype=mu_dtype,
                    generator=torch_rng,
                )
                for i, (mem, std) in enumerate(zip(memory_bank, stds)):
                    mem["mu"] += noise_batch[i:i + 1] * (std * scaled_std)

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
                for mem, std in zip(memory_bank, stds):
                    score = compute_score(probe, mem["mu"], std, metric)
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


# ── wrapper matching run_experiment_scores_prior API ──────────────────

def run_experiment_scores_2d(debug=False, seed=0, **kwargs):
    """Convenience wrapper matching ``run_experiment_scores_prior`` API.

    Builds a ``ThreeRegimeNoise`` schedule from keyword arguments and
    forwards to ``run_model_core_2d``.
    """
    schedule = ThreeRegimeNoise(
        sigma0=kwargs.get("sigma0", 0.1),
        sigma1=kwargs.get("sigma1", 0.0),
        sigma2=kwargs.get("sigma2", 0.0),
        t_step=kwargs.get("t_step", 5),
    )
    return run_model_core_2d(
        sigma0=kwargs["sigma0"],
        X0=kwargs["X0"],
        name_to_idx=kwargs["name_to_idx"],
        experiment_list=kwargs["experiment_list"],
        score_model=kwargs["score_model"],
        drift_step_size=kwargs.get("drift_step_size", 0.0),
        metric=kwargs.get("metric", "mahalanobis"),
        noise_schedule=schedule,
        debug=debug,
        seed=seed,
    )


# ── high-level ISI sweep ─────────────────────────────────────────────

def run_2d_isi_sweep(
    sigma0,
    sigma1,
    sigma2,
    drift_step_size,
    score_model,
    X0,
    name_to_idx,
    stimulus_pool,
    t_step=5,
    isi_values=(0, 1, 2, 4, 8, 16, 32, 64),
    n_experiments=20,
    k_stimuli=10,
    n_mc=16,
    seed=42,
):
    """Run a full ISI sweep and return d' (mean ± SEM) per ISI.

    Parameters
    ----------
    sigma0 : float
        Encoding noise.
    sigma1, sigma2 : float
        Per-step drift noise (short / long range).
    drift_step_size : float
        Prior-driven drift magnitude.
    score_model : ScoreAdapter2D
    X0, name_to_idx, stimulus_pool : as from ``make_2d_grid_stimuli``.
    t_step : int
        Age threshold between sigma1 and sigma2 regimes.
    isi_values : tuple of int
        ISI conditions to evaluate.
    n_experiments : int
        Toy experiments per ISI.
    k_stimuli : int
        Stimuli per experiment.
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
    schedule = ThreeRegimeNoise(sigma0, sigma1, sigma2, t_step)
    results_isi = []
    results_dprime_mean = []
    results_dprime_sem = []
    results_auroc = []
    raw_runs = {}

    for isi in isi_values:
        exp_list = make_toy_experiment_list(
            stimulus_pool, isi=isi,
            n_experiments=n_experiments, k_stimuli=k_stimuli,
            seed=seed + isi * 100,
        )

        all_hits, all_fas = [], []
        for rep in range(n_mc):
            run_out = run_model_core_2d(
                sigma0=sigma0,
                X0=X0,
                name_to_idx=name_to_idx,
                experiment_list=exp_list,
                score_model=score_model,
                drift_step_size=drift_step_size,
                metric="mahalanobis",
                noise_schedule=schedule,
                seed=seed * 10_000 + isi * 1000 + rep,
            )
            all_hits.extend(run_out["hits"])
            all_fas.extend(run_out["fas"])

        hits_arr = np.asarray(all_hits, float)
        fas_arr = np.asarray(all_fas, float)

        # Compute AUROC → d'
        roc_res = roc_from_arrays(hits_arr, fas_arr, score_type="distance")
        if roc_res is not None:
            _, _, auc_val = roc_res
            dp = auroc_to_dprime(auc_val)
        else:
            auc_val = np.nan
            dp = np.nan

        # Build a synthetic run_data for bootstrap
        isi_hit_dists = {isi: [(h, 0) for h in hits_arr]}
        run_data = {
            "isi_hit_dists": isi_hit_dists,
            "fas": fas_arr,
            "score_type": "distance",
        }
        mean_dp, sem_dp = bootstrap_dprime_ci(run_data, isi, n_boot=200)

        results_isi.append(isi)
        results_dprime_mean.append(float(mean_dp) if np.isfinite(mean_dp) else dp)
        results_dprime_sem.append(float(sem_dp) if np.isfinite(sem_dp) else 0.0)
        results_auroc.append(float(auc_val))
        raw_runs[isi] = run_data

    return {
        "isi_values": results_isi,
        "dprime_mean": results_dprime_mean,
        "dprime_sem": results_dprime_sem,
        "auroc": results_auroc,
        "raw_runs": raw_runs,
    }
