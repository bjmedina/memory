"""
utls.runners_prior — Unified prior-guided drift-diffusion simulation engine.

Generalises the 2D sandbox runner (``runners_2d.run_model_core_2d_vec``) to
work in **any** dimensionality.  Drop-in compatible with:

  * ``ScoreAdapter2D``  (2D analytic GMM sandbox)
  * ``ScoreFunction``   (learned 256-PC score on real auditory textures)

The model has three free parameters (four with a noise schedule):

  * σ₀  — encoding noise (applied once at memory insertion)
  * σ   — diffusive noise (constant per-step Langevin noise)
  * η   — prior-driven drift step size
  * noise_schedule — optional callable mapping memory age → σ(age),
    replacing the constant σ with a decaying schedule

This module is the single source of truth for prior-guided simulations,
regardless of stimulus dimensionality.
"""

import numpy as np
import torch
from collections import defaultdict

from utls.runners_v2 import compute_score


# ── batched scoring ───────────────────────────────────────────────────

def compute_scores_batched(probe, mu_bank, n_mem, sigma, metric):
    """Vectorised scoring: probe [1, D] vs mu_bank[:n_mem] [n_mem, D].

    Returns a 1-D tensor of shape [n_mem].
    """
    mem = mu_bank[:n_mem]                                       # [n_mem, D]

    if metric == "cosine":
        probe_n = probe / (probe.norm() + 1e-12)               # [1, D]
        mem_n = mem / (mem.norm(dim=1, keepdim=True) + 1e-12)   # [n_mem, D]
        cos_sims = (probe_n * mem_n).sum(dim=1)                 # [n_mem]
        return 1.0 - cos_sims

    diff = probe - mem                                          # [n_mem, D]
    sqdist = (diff ** 2).sum(dim=1)                             # [n_mem]

    if metric == "euclidean":
        return sqdist.sqrt()
    elif metric == "manhattan":
        return diff.abs().sum(dim=1)
    elif metric == "mahalanobis":
        return sqdist.sqrt() / sigma
    elif metric == "loglikelihood":
        D_dim = probe.shape[1]
        var = sigma ** 2
        return (
            -0.5 * D_dim * np.log(2 * np.pi)
            - D_dim * np.log(sigma)
            - 0.5 * (sqdist / var)
        )
    else:
        raise ValueError(f"Unknown metric '{metric}'")


# ── core simulation engine ────────────────────────────────────────────

def run_model_core_prior(
    sigma0,
    sigma,
    *,
    X0,
    name_to_idx,
    experiment_list,
    score_model,
    drift_step_size=0.0,
    noise_schedule=None,
    metric="cosine",
    seed=0,
    torch_rng=None,
    return_item_scores=False,
    return_trial_log=False,
):
    """Prior-guided drift-diffusion memory simulation (any dimensionality).

    This is the unified engine for the prior-guided model.  It is a
    generalisation of ``runners_2d.run_model_core_2d_vec`` that works with
    stimuli of *any* dimensionality D — from the 2D GMM sandbox to the
    full 256-PC auditory texture space.

    Parameters
    ----------
    sigma0 : float
        Encoding noise magnitude (applied once at memory insertion).
    sigma : float
        Diffusive noise magnitude (constant per-step during Langevin dynamics).
    X0 : Tensor [N, D]
        Stimulus embeddings (D can be 2, 256, or any other dimensionality).
    name_to_idx : dict
        Stimulus name → row index in *X0*.
    experiment_list : list[list[str]]
        Sequences of stimulus names.
    score_model : object
        Any object with a ``.forward(x)`` method that accepts [B, D] tensors
        and returns [B, D] (or [B, 1, 1, D]) score vectors.
        Works with both ``ScoreAdapter2D`` and ``ScoreFunction``.
    drift_step_size : float
        Magnitude of prior-driven drift per trial (η in the paper).
    noise_schedule : callable or None
        If provided, a callable mapping memory age (int or Tensor of ints)
        to per-step noise σ(age).  Overrides the constant ``sigma`` for
        diffusive noise.  See ``utls.noise_schedules`` for implementations.
        When None (default), constant ``sigma`` is used (standard M2).
    metric : str
        Distance metric for decision.
    seed : int
        Random seed.
    torch_rng : torch.Generator or None

    return_item_scores : bool
        If True, also return ``item_hits``/``item_fas`` dictionaries containing
        per-stimulus score lists.
    return_trial_log : bool
        If True, return ``trial_log`` with one row per scored trial
        (repeat/foil), including ISI and raw score.

    Returns
    -------
    dict
        Keys: ``hits``, ``fas``, ``isi_hit_dists``, ``fa_by_t``,
        ``T_max``, ``metric``, ``score_type``.
        Optional keys: ``item_hits``, ``item_fas``, ``trial_log``.
    """
    if torch_rng is None:
        torch_rng = torch.Generator(device=X0.device)
        torch_rng.manual_seed(seed)

    D = X0.shape[1]

    dim_std = X0.std(0, unbiased=True)
    rms_std = torch.sqrt(torch.mean(dim_std ** 2)).item()
    scaled_std = dim_std / rms_std                              # [D]

    hit_scores, fa_scores = [], []
    isi_hit_dists = defaultdict(list)
    T_max = max((len(seq) for seq in experiment_list), default=0)
    fa_by_t = [[] for _ in range(T_max)]
    idx_to_name = {v: k for k, v in name_to_idx.items()}
    item_hits = defaultdict(list)
    item_fas = defaultdict(list)
    trial_log = []

    for seq_i, seq in enumerate(experiment_list):
        if not seq:
            continue

        seq_idx = [name_to_idx[f] for f in seq]
        seq_len = len(seq_idx)

        # Pre-allocated memory bank: [seq_len, D]
        mu_bank = torch.zeros(seq_len, D, device=X0.device, dtype=X0.dtype)
        insertion_times = torch.zeros(seq_len, device=X0.device, dtype=torch.long)
        n_mem = 0
        seen, last_seen = set(), {}

        for t, incoming in enumerate(seq_idx, start=1):
            probe = X0[incoming].view(1, -1)                    # [1, D]

            # ---------- UPDATE + SCORE MEMORIES ----------
            if n_mem > 0:
                # diffusive noise (batched)
                noise = torch.randn(
                    n_mem, D,
                    device=X0.device, dtype=X0.dtype,
                    generator=torch_rng,
                )
                if noise_schedule is not None:
                    ages = t - insertion_times[:n_mem]          # [n_mem]
                    sigmas = noise_schedule(ages)               # [n_mem] tensor
                    if not isinstance(sigmas, torch.Tensor):
                        sigmas = torch.tensor(sigmas, device=X0.device, dtype=X0.dtype)
                    mu_bank[:n_mem] += noise * (sigmas[:, None] * scaled_std[None, :])
                else:
                    mu_bank[:n_mem] += noise * (sigma * scaled_std)

                # prior-driven drift (batched)
                if drift_step_size > 0:
                    with torch.no_grad():
                        drift = score_model.forward(mu_bank[:n_mem])
                    # ScoreFunction returns [B, 1, 1, D]; flatten to [B, D]
                    if drift.dim() == 4:
                        drift = drift.reshape(drift.shape[0], -1)
                    elif drift.dim() == 1:
                        drift = drift.unsqueeze(0)
                    mu_bank[:n_mem] += drift_step_size * drift

                # decision scores (batched)
                scores_t = compute_scores_batched(
                    probe, mu_bank, n_mem, sigma, metric,
                )
                if metric == "loglikelihood":
                    score_val = scores_t.max().item()
                else:
                    score_val = scores_t.min().item()
            else:
                score_val = None

            # ---------- DECISION STEP ----------
            if score_val is not None:
                is_repeat = incoming in seen
                stim_name = idx_to_name[incoming]
                if is_repeat:
                    hit_scores.append(score_val)
                    isi = t - last_seen[incoming]
                    isi_hit_dists[isi].append((score_val, t))
                    if return_item_scores and isi > 1:
                        item_hits[stim_name].append(score_val)
                    if return_trial_log:
                        trial_log.append({
                            "sequence_index": seq_i,
                            "trial_index": t,
                            "stimulus": stim_name,
                            "is_repeat": True,
                            "isi": isi - 1,
                            "score": score_val,
                        })
                else:
                    fa_scores.append(score_val)
                    fa_by_t[t - 1].append(score_val)
                    if return_item_scores:
                        item_fas[stim_name].append(score_val)
                    if return_trial_log:
                        trial_log.append({
                            "sequence_index": seq_i,
                            "trial_index": t,
                            "stimulus": stim_name,
                            "is_repeat": False,
                            "isi": np.nan,
                            "score": score_val,
                        })

            # ---------- STORE NEW MEMORY ----------
            base = X0[incoming].clone()
            noise_enc = torch.randn(
                base.shape, device=base.device, dtype=base.dtype,
                generator=torch_rng,
            )
            mu_bank[n_mem] = base + noise_enc * (sigma0 * dim_std)
            insertion_times[n_mem] = t
            n_mem += 1
            seen.add(incoming)
            last_seen[incoming] = t

    out = {
        "hits": np.array(hit_scores),
        "fas": np.array(fa_scores),
        "isi_hit_dists": dict(isi_hit_dists),
        "fa_by_t": fa_by_t,
        "T_max": T_max,
        "metric": metric,
        "score_type": "likelihood" if metric == "loglikelihood" else "distance",
    }
    if return_item_scores:
        out["item_hits"] = dict(item_hits)
        out["item_fas"] = dict(item_fas)
    if return_trial_log:
        out["trial_log"] = trial_log
    return out
