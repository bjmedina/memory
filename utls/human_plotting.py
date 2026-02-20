"""
Plotting functions for human auditory memory analysis.

All functions accept matplotlib axes or create their own figures,
and optionally save to disk.
"""

import os
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# ── d-prime vs ISI curves ────────────────────────────────────────────

def plot_dprime_vs_isi(
    analysis_results,
    labels=None,
    colors=None,
    title=None,
    ylim=(0, 3.5),
    save_path=None,
    ax=None,
):
    """
    Plot one or more d' vs ISI curves with bootstrap error bars.

    Parameters
    ----------
    analysis_results : dict or list[dict]
        Output(s) of run_analysis(). Each must have keys:
        isis, dprime, boot (with 'sem'), N.
    labels : list[str] or None
        Legend labels. If None, uses "Condition i (N=...)".
    colors : list[str] or None
        Colors for each curve.
    title : str or None
    ylim : tuple or None
    save_path : str or None
    ax : matplotlib Axes or None
    """
    if isinstance(analysis_results, dict):
        analysis_results = [analysis_results]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))

    default_colors = ["g", "b", "orange", "red", "purple", "brown", "pink"]
    if colors is None:
        colors = default_colors[: len(analysis_results)]
    if labels is None:
        labels = [f"Condition {i} (N={r['N']})" for i, r in enumerate(analysis_results)]

    for out, color, label in zip(analysis_results, colors, labels):
        isis = out["isis"]
        ax.errorbar(
            isis,
            out["dprime"],
            yerr=out["boot"]["sem"],
            fmt="o-",
            color=color,
            capsize=4,
            linewidth=2,
            markersize=6,
            label=label,
        )

    ax.set_xticks(analysis_results[0]["isis"])
    ax.set_xticklabels(analysis_results[0]["isis"])
    if ylim:
        ax.set_ylim(ylim)
    ax.set_xlabel("Inter-Stimulus Interval (ISI)")
    ax.set_ylabel("d' (Sensitivity)")
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend()

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_hit_and_fa_rates(
    analysis_results,
    labels=None,
    colors=None,
    title=None,
    save_path=None,
    ax=None,
):
    """
    Plot hit rates vs ISI with false-alarm rate baselines.

    Parameters
    ----------
    analysis_results : dict or list[dict]
        Output(s) of run_analysis(). Must have: hrs, fa, N.
    """
    if isinstance(analysis_results, dict):
        analysis_results = [analysis_results]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))

    default_colors = ["g", "b", "orange", "red", "purple"]
    if colors is None:
        colors = default_colors[: len(analysis_results)]
    if labels is None:
        labels = [f"Condition {i} (N={r['N']})" for i, r in enumerate(analysis_results)]

    for out, color, label in zip(analysis_results, colors, labels):
        hrs = out["hrs"]
        # Skip ISI=-1 if present
        hrs_clean = hrs[hrs.index != -1]
        isis = np.array(list(hrs_clean.keys()))
        hit_rates = np.array([hrs_clean[k] for k in isis])

        ax.plot(isis, hit_rates, "o-", color=color, linewidth=2, markersize=6,
                label=f"{label} Hit Rate")
        ax.axhline(y=out["fa"], color=color, linestyle="--", linewidth=2,
                    label=f"{label} FA={out['fa']:.3f}")

    ax.set_ylim(0, 1)
    ax.set_xlabel("Inter-Stimulus Interval (ISI)")
    ax.set_ylabel("Rate")
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", ls="--", alpha=0.4)
    ax.legend(fontsize=8)

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


# ── split-half reliability histogram ─────────────────────────────────

def plot_split_half_histogram(
    reliability_result,
    title=None,
    save_path=None,
    ax=None,
):
    """
    Histogram of split-half d' curve correlations.

    Parameters
    ----------
    reliability_result : dict
        Output of split_half_reliability(). Has 'correlations', 'mean_r', 'ci'.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))

    corrs = reliability_result["correlations"]
    ax.hist(corrs, bins=40, edgecolor="black", alpha=0.7)
    ax.axvline(reliability_result["mean_r"], color="red", ls="--",
               label=f"mean r = {reliability_result['mean_r']:.3f}")
    ax.set_xlabel("Correlation (split-half)")
    ax.set_ylabel("Count")
    if title:
        ax.set_title(title)
    ax.legend()

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


# ── inter-response timing ────────────────────────────────────────────

def plot_inter_response_times(
    irts,
    n_std=2,
    title=None,
    save_path=None,
    ax=None,
):
    """
    Histogram of inter-response time intervals.

    Parameters
    ----------
    irts : array
        Inter-response times in ms (from inter_response_times()).
    n_std : int
        Number of stds above mean for x-axis limit.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))

    xlim = np.mean(irts) + n_std * np.std(irts)
    ax.hist(irts, bins=1000, edgecolor="none")
    ax.set_xlabel("Inter-response time (ms)")
    ax.set_ylabel("Count")
    ax.set_xlim([0, xlim])
    if title is None:
        title = f"Mean IRT: {np.mean(irts):.0f} ms"
    ax.set_title(title)

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


# ── random-split replication ─────────────────────────────────────────

def plot_random_split(
    df,
    run_analysis_fn,
    title=None,
    seed=0,
    save_path=None,
    ax=None,
):
    """
    Split subjects randomly into two halves and overlay their d' curves.

    Parameters
    ----------
    df : DataFrame
        Per-subject-per-ISI data.
    run_analysis_fn : callable
        The run_analysis function.
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))

    subjects = df["subject"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)
    half = len(subjects) // 2

    xA = df[df["subject"].isin(subjects[:half])].copy()
    xB = df[df["subject"].isin(subjects[half:])].copy()

    outA = run_analysis_fn(xA)
    outB = run_analysis_fn(xB)

    ax.errorbar(outA["isis"], outA["dprime"], yerr=outA["boot"]["sem"],
                fmt="o-", capsize=3, linewidth=2, markersize=6,
                label=f"Split A (N={outA['N']})")
    ax.errorbar(outB["isis"], outB["dprime"], yerr=outB["boot"]["sem"],
                fmt="s-", capsize=3, linewidth=2, markersize=6,
                label=f"Split B (N={outB['N']})")

    ax.set_xticks(outA["isis"])
    ax.set_xticklabels(outA["isis"])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Inter-Stimulus Interval (ISI)")
    ax.set_ylabel("d' (Sensitivity)")
    if title:
        ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.4)
    ax.legend()

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


# ── stimulus frequency histogram ─────────────────────────────────────

def plot_stimulus_frequency(
    exps,
    isis=None,
    title=None,
    save_path=None,
    ax=None,
):
    """
    Overlay stimulus frequency curves across ISIs.

    Parameters
    ----------
    exps : list[DataFrame]
        Raw experiment DataFrames.
    isis : list[int]
        ISIs to show.
    """
    if isis is None:
        isis = [0, 1, 2, 4, 8, 16, 32, 64]

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Collect all stimulus IDs
    all_ids = set()
    for exp in exps:
        for i in exp.index:
            stim = exp.loc[i].stimulus.split("/")[-1]
            nums = re.findall(r"\d+", stim)
            if nums:
                all_ids.add(int(nums[-1]))
    all_ids = np.array(sorted(all_ids))

    for j in isis:
        freq = defaultdict(int)
        for exp in exps:
            for i in exp.index:
                if exp.loc[i].isi == j:
                    stim = exp.loc[i].stimulus.split("/")[-1]
                    nums = re.findall(r"\d+", stim)
                    if nums:
                        freq[int(nums[-1])] += 1
        counts = np.array([freq[sid] for sid in all_ids])
        ax.plot(all_ids, counts, label=f"ISI {j}", linewidth=2, alpha=0.5)

    ax.legend()
    ax.set_xlabel("Stimulus ID")
    ax.set_ylabel("Count")
    ax.set_ylim(bottom=0)
    if len(all_ids) > 10:
        ax.set_xticks(all_ids[::10])
    if title:
        ax.set_title(title)

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


# ── cross-experiment itemwise scatter ────────────────────────────────

def plot_itemwise_scatter(
    rates_a,
    rates_b,
    r_value,
    kind="hits",
    xlabel="Short ISI",
    ylabel="Long ISI",
    title=None,
    noise_ceiling=None,
    color=None,
    jitter=0.01,
    save_path=None,
    ax=None,
):
    """
    Scatter plot of itemwise rates between two conditions.

    Parameters
    ----------
    rates_a, rates_b : array
        Aligned itemwise rates.
    r_value : float
        Correlation between them.
    kind : str
        'hits' or 'false_alarms' (for display).
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 6))

    if color is None:
        color = "green" if kind == "hits" else "red"

    x = rates_a + np.random.normal(0, jitter, size=len(rates_a))
    y = rates_b + np.random.normal(0, jitter, size=len(rates_b))

    ax.scatter(x, y, color=color, alpha=0.7, edgecolor="k", s=20)
    ax.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    metric = "Hit Rate" if kind == "hits" else "False-Alarm Rate"
    ax.set_xlabel(f"{xlabel} {metric}")
    ax.set_ylabel(f"{ylabel} {metric}")

    txt = f"r = {r_value:.2f}\nN = {len(rates_a)}"
    if noise_ceiling is not None:
        txt += f"\nCeiling = {noise_ceiling:.2f}"
    ax.text(0.05, 0.9, txt, transform=ax.transAxes, fontsize=10,
            verticalalignment="top")

    if title:
        ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.4)

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


# ── power curve plots ────────────────────────────────────────────────

def plot_power_curve(
    power_result,
    kind="hits",
    title=None,
    save_path=None,
    ax=None,
):
    """
    Plot a power analysis curve (correlation vs sample size).

    Parameters
    ----------
    power_result : dict
        Output of itemwise_power_analysis() or dprime_curve_power_analysis().
        Must have: Ns, mean/r_mean, and either ci or r_ci_low/r_ci_high.
    kind : str
        'hits' or 'false_alarms' (for display).
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 5))

    Ns = power_result["Ns"]
    # Handle both naming conventions
    means = power_result.get("mean", power_result.get("r_mean"))

    if "ci" in power_result and power_result["ci"].ndim == 2:
        ci_low = power_result["ci"][:, 0]
        ci_high = power_result["ci"][:, 1]
    else:
        ci_low = power_result.get("r_ci_low", means - 0.1)
        ci_high = power_result.get("r_ci_high", means + 0.1)

    ax.plot(Ns, means, "o-", linewidth=2)
    ax.fill_between(Ns, ci_low, ci_high, alpha=0.2)
    ax.set_xlabel("Participants per Condition")
    ax.set_ylabel(f"Correlation ({kind})")
    ax.set_ylim(0, 1)
    ax.grid(True, ls="--", alpha=0.4)
    if title:
        ax.set_title(title)

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
