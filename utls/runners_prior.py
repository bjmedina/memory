"""
utls.runners_prior — Unified prior-guided drift-diffusion simulation engine.

Generalises the 2D sandbox runner (``runners_2d.run_model_core_2d_vec``) to
work in **any** dimensionality.  Drop-in compatible with:

  * ``ScoreAdapter2D``  (2D analytic GMM sandbox)
  * ``ScoreFunction``   (learned 256-PC score on real auditory textures)

The model has three free parameters:

  * σ₀  — encoding noise (applied once at memory insertion)
  * σ   — diffusive noise (constant per-step Langevin noise)
  * η   — prior-driven drift step size

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
    metric="cosine",
    seed=0,
    torch_rng=None,
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
    metric : str
        Distance metric for decision.
    seed : int
        Random seed.
    torch_rng : torch.Generator or None

    Returns
    -------
    dict
        Keys: ``hits``, ``fas``, ``isi_hit_dists``, ``fa_by_t``,
        ``T_max``, ``metric``, ``score_type``.
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

    for seq in experiment_list:
        if not seq:
            continue

        seq_idx = [name_to_idx[f] for f in seq]
        seq_len = len(seq_idx)

        # Pre-allocated memory bank: [seq_len, D]
        mu_bank = torch.zeros(seq_len, D, device=X0.device, dtype=X0.dtype)
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
                if is_repeat:
                    hit_scores.append(score_val)
                    isi = t - last_seen[incoming]
                    isi_hit_dists[isi].append((score_val, t))
                else:
                    fa_scores.append(score_val)
                    fa_by_t[t - 1].append(score_val)

            # ---------- STORE NEW MEMORY ----------
            base = X0[incoming].clone()
            noise_enc = torch.randn(
                base.shape, device=base.device, dtype=base.dtype,
                generator=torch_rng,
            )
            mu_bank[n_mem] = base + noise_enc * (sigma0 * dim_std)
            n_mem += 1
            seen.add(incoming)
            last_seen[incoming] = t

    return {
        "hits": np.array(hit_scores),
        "fas": np.array(fa_scores),
        "isi_hit_dists": dict(isi_hit_dists),
        "fa_by_t": fa_by_t,
        "T_max": T_max,
        "metric": metric,
        "score_type": "likelihood" if metric == "loglikelihood" else "distance",
    }
