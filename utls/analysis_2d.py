"""
utls.analysis_2d — analysis and plotting for the 2D guided-drift sandbox.

Provides d' curves, item susceptibility analysis, prior-mismatch
benchmarks, and publication-ready plotting helpers.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from collections import defaultdict

from utls.runners_2d import run_model_core_2d, run_2d_isi_sweep
from utls.toy_experiments import make_high_diversity_sequences
from utls.roc_utils import roc_from_arrays
from utls.analysis_helpers import auroc_to_dprime


# ── d' curve extraction ──────────────────────────────────────────────

def dprime_by_isi_curve(sweep_results):
    """Extract arrays from ``run_2d_isi_sweep`` output for plotting.

    Returns
    -------
    isis : np.ndarray
    dprimes : np.ndarray
    sems : np.ndarray
    """
    return (
        np.array(sweep_results["isi_values"]),
        np.array(sweep_results["dprime_mean"]),
        np.array(sweep_results["dprime_sem"]),
    )


# ── item susceptibility ──────────────────────────────────────────────

def item_susceptibility_analysis(
    score_model,
    X0,
    name_to_idx,
    stimulus_pool,
    descriptors_df,
    sigma0,
    sigma,
    drift_step_size,
    metric="cosine",
    isi_values=(1, 4, 16, 64),
    n_mc=32,
    seed=42,
):
    """Per-item error analysis correlated with geometry descriptors.

    Runs Monte Carlo simulations with interleaved multi-ISI sequences,
    computes per-item presentation counts, and merges with geometry
    descriptors.

    Parameters
    ----------
    sigma0 : float
        Encoding noise.
    sigma : float
        Diffusive noise (constant per step).
    metric : str
        Distance metric (default ``"cosine"``).

    Returns
    -------
    pd.DataFrame
        One row per (point, ISI) with columns: ``point_id``, ``isi``,
        ``n_presentations``, plus all geometry columns from *descriptors_df*.
    """
    # Generate interleaved multi-ISI sequences
    max_isi = max(isi_values)
    seq_length = max(max_isi + 5, 42)
    # Round up to nearest multiple of 3
    seq_length = ((seq_length + 2) // 3) * 3

    exp_list, isi_keys = make_high_diversity_sequences(
        stimulus_pool=stimulus_pool,
        isi_values=list(isi_values),
        n_sequences=10,
        length=seq_length,
        min_pairs_per_isi=3,
        seed=seed,
    )

    rows = []

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

    # For item-level: count presentations per item across experiments
    for isi in isi_values:
        item_count = defaultdict(int)
        for seq in exp_list:
            seen = {}
            for pos, stim in enumerate(seq):
                if stim in seen:
                    actual_isi = pos - seen[stim] - 1
                    if actual_isi == isi:
                        item_count[stim] += 1
                else:
                    seen[stim] = pos

        for pt_id in stimulus_pool:
            n_pres = item_count.get(pt_id, 0)
            if n_pres == 0:
                continue
            rows.append({
                "point_id": pt_id,
                "isi": isi,
                "n_presentations": n_pres * n_mc,
            })

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df = df.merge(descriptors_df, on="point_id", how="left")
    return df


# ── prior mismatch benchmark ─────────────────────────────────────────

def prior_mismatch_benchmark(
    matched_gmm,
    mismatched_gmm,
    X0,
    name_to_idx,
    stimulus_pool,
    sigma0,
    sigma,
    drift_step_size,
    metric="cosine",
    isi_values=(0, 1, 2, 4, 8, 16, 32, 64),
    n_mc=16,
    seed=42,
):
    """Compare d' curves under matched vs mismatched prior.

    Parameters
    ----------
    matched_gmm, mismatched_gmm : AnalyticGMM2D
        Correct and incorrect prior, respectively.
    sigma0 : float
        Encoding noise.
    sigma : float
        Diffusive noise (constant per step).
    metric : str
        Distance metric (default ``"cosine"``).

    Returns
    -------
    dict with keys ``"matched"`` and ``"mismatched"``, each a
    ``run_2d_isi_sweep`` result dict.
    """
    from src.model.score_adapter_2d import ScoreAdapter2D

    results = {}
    for label, gmm in [("matched", matched_gmm), ("mismatched", mismatched_gmm)]:
        adapter = ScoreAdapter2D(gmm, normalize=True)
        sweep = run_2d_isi_sweep(
            sigma0=sigma0,
            sigma=sigma,
            drift_step_size=drift_step_size,
            score_model=adapter,
            X0=X0,
            name_to_idx=name_to_idx,
            stimulus_pool=stimulus_pool,
            isi_values=isi_values,
            metric=metric,
            n_mc=n_mc,
            seed=seed,
        )
        results[label] = sweep

    return results


# ── plotting helpers ──────────────────────────────────────────────────

def plot_prior_with_score_field(gmm, X0=None, grid_n=30, ax=None, title=None):
    """Contour plot of GMM density + quiver plot of score field.

    Parameters
    ----------
    gmm : AnalyticGMM2D
    X0 : Tensor [N, 2] or None
        If given, overlay stimulus points.
    grid_n : int
        Resolution of contour / quiver grid.
    ax : matplotlib Axes or None
    title : str or None
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    xs = np.linspace(-5, 5, grid_n)
    ys = np.linspace(-5, 5, grid_n)
    Xg, Yg = np.meshgrid(xs, ys)
    pts = torch.tensor(
        np.stack([Xg.ravel(), Yg.ravel()], axis=1), dtype=torch.float64
    )

    log_p = gmm.log_prob(pts).numpy().reshape(grid_n, grid_n)
    score_vecs = gmm.score(pts).numpy()
    U = score_vecs[:, 0].reshape(grid_n, grid_n)
    V = score_vecs[:, 1].reshape(grid_n, grid_n)

    # Density contours
    ax.contourf(Xg, Yg, np.exp(log_p), levels=20, cmap="Blues", alpha=0.7)
    ax.contour(Xg, Yg, np.exp(log_p), levels=8, colors="steelblue",
               linewidths=0.5, alpha=0.6)

    # Score field (subsample for readability)
    skip = max(1, grid_n // 15)
    ax.quiver(
        Xg[::skip, ::skip], Yg[::skip, ::skip],
        U[::skip, ::skip], V[::skip, ::skip],
        color="darkred", alpha=0.6, scale=25, width=0.004,
    )

    # Mixture means
    means = gmm.means.numpy()
    ax.scatter(means[:, 0], means[:, 1], marker="x", s=100, c="black",
               linewidths=2, zorder=10, label="means")

    # Stimulus points
    if X0 is not None:
        x_np = X0.numpy() if isinstance(X0, torch.Tensor) else X0
        ax.scatter(x_np[:, 0], x_np[:, 1], s=15, c="orange", edgecolors="k",
                   linewidths=0.3, zorder=5, label="stimuli")

    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left")
    if title:
        ax.set_title(title)
    return ax


def plot_dprime_curves(results_dict, title="", ax=None):
    """Plot d' vs ISI for one or more conditions.

    Parameters
    ----------
    results_dict : dict[str, dict]
        Label → ``run_2d_isi_sweep`` result.
    title : str
    ax : Axes or None
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4.5))

    for label, res in results_dict.items():
        isis = np.array(res["isi_values"], float)
        dp = np.array(res["dprime_mean"])
        sem = np.array(res["dprime_sem"])

        # Use ISI+0.5 for log-scale x axis (avoid log(0))
        x = isis.copy()
        x[x == 0] = 0.5

        line, = ax.plot(x, dp, "o-", label=label, markersize=5)
        ax.fill_between(x, dp - sem, dp + sem, alpha=0.2,
                        color=line.get_color())

    ax.set_xscale("log")
    ax.set_xlabel("ISI")
    ax.set_ylabel("d'")
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    return ax


def plot_item_susceptibility(suscept_df, predictor_cols=None, ax=None):
    """Scatter plots: per-item geometry descriptors vs ISI presence.

    Parameters
    ----------
    suscept_df : pd.DataFrame
        From :func:`item_susceptibility_analysis`.
    predictor_cols : list[str] or None
        Geometry columns to plot. Defaults to a standard set.
    """
    if predictor_cols is None:
        predictor_cols = [
            "log_density", "score_norm",
            "dist_to_nearest_mean", "posterior_entropy",
        ]

    # Only use cols that exist
    predictor_cols = [c for c in predictor_cols if c in suscept_df.columns]
    n_cols = len(predictor_cols)
    if n_cols == 0:
        return None

    isis = sorted(suscept_df["isi"].unique())

    fig, axes = plt.subplots(
        len(isis), n_cols, figsize=(4 * n_cols, 3.5 * len(isis)),
        squeeze=False,
    )

    for i, isi in enumerate(isis):
        sub = suscept_df[suscept_df["isi"] == isi]
        for j, col in enumerate(predictor_cols):
            ax = axes[i, j]
            ax.scatter(sub[col], sub["n_presentations"], s=15, alpha=0.6)
            ax.set_xlabel(col)
            if j == 0:
                ax.set_ylabel(f"ISI={isi}\nn_pres")
            ax.grid(alpha=0.2)

    fig.suptitle("Item Susceptibility by Geometry", y=1.02, fontsize=13)
    fig.tight_layout()
    return fig
