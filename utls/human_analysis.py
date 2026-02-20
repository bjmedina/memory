"""
Core analysis functions for human auditory memory experiments.

Provides d-prime computation, bootstrapping, split-half reliability,
and power analysis for forgetting curve data.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, pearsonr, spearmanr


# ── d-prime helpers ──────────────────────────────────────────────────

def clip_rate(p, eps=1e-2):
    """Clip a probability away from 0 and 1."""
    return np.clip(p, eps, 1 - eps)


def dprime_from_rates(hit, fa, eps=1e-2):
    """Compute d' from hit and false-alarm rates."""
    return norm.ppf(clip_rate(hit, eps)) - norm.ppf(clip_rate(fa, eps))


def aprime_from_rates(hit, fa):
    """
    Compute nonparametric A' (Snodgrass & Corwin, 1988).
    Accepts scalar or array inputs.
    """
    hit = np.asarray(hit, dtype=float)
    fa = np.asarray(fa, dtype=float)
    eps = 1e-9
    hit = np.clip(hit, eps, 1 - eps)
    fa = np.clip(fa, eps, 1 - eps)

    A = np.zeros_like(hit)
    mask = hit >= fa
    A[mask] = 0.5 + ((hit[mask] - fa[mask]) * (1 + hit[mask] - fa[mask])) / (
        4 * hit[mask] * (1 - fa[mask])
    )
    A[~mask] = 0.5 - ((fa[~mask] - hit[~mask]) * (1 + fa[~mask] - hit[~mask])) / (
        4 * fa[~mask] * (1 - hit[~mask])
    )
    return A


# ── population-level curve computation ───────────────────────────────

def population_fa_rate(df):
    """Mean false-alarm rate across subjects."""
    return df["fa_rate"].mean()


def population_hit_rates_by_isi(df):
    """Mean hit rate per ISI across subjects. Returns Series indexed by ISI."""
    return df.groupby("isi")["hit_rate"].mean().sort_index()


def compute_dprime_curve(df, eps=1e-2):
    """
    Compute the population d' curve from a per-subject-per-ISI DataFrame.

    Parameters
    ----------
    df : DataFrame
        Must have columns: subject, isi, hit_rate, fa_rate.

    Returns
    -------
    isis : ndarray
        ISI values.
    dprime : ndarray
        d' at each ISI.
    """
    fa = population_fa_rate(df)
    hrs = population_hit_rates_by_isi(df)
    isis = hrs.index.to_numpy()
    dp = np.array([dprime_from_rates(h, fa, eps=eps) for h in hrs.values])
    return isis, dp


def compute_dprime_for_subjects(df, subject_list, eps=1e-2):
    """Compute d' curve for a subset of subjects."""
    return compute_dprime_curve(df[df["subject"].isin(subject_list)], eps=eps)


# ── run_analysis: single-call summary ────────────────────────────────

def run_analysis(df, n_boot=5000, seed=0):
    """
    Full population-level analysis pipeline.

    Parameters
    ----------
    df : DataFrame
        Per-subject-per-ISI summary (from recompute_dprime_by_isi_per_subject).
    n_boot : int
        Bootstrap resamples for CIs.

    Returns
    -------
    dict with keys: isis, dprime, boot, hrs, fa, N.
    """
    df_clean = df[df["isi"] != -1].copy()
    isis, dp = compute_dprime_curve(df_clean)
    boot = bootstrap_dprime(df_clean, n_boot=n_boot, seed=seed)
    N = df_clean["subject"].nunique()
    hrs = population_hit_rates_by_isi(df)
    fa = population_fa_rate(df)
    return {
        "isis": isis,
        "dprime": dp,
        "boot": boot,
        "hrs": hrs,
        "fa": fa,
        "N": N,
    }


# ── bootstrapping ────────────────────────────────────────────────────

def bootstrap_dprime(df, n_boot=5000, seed=0):
    """
    Bootstrap confidence intervals on the population d' curve.

    Resamples subjects with replacement, recomputes d' each time.

    Returns
    -------
    dict with: isis, mean, sem, ci_low, ci_high, boot_matrix.
    """
    rng = np.random.default_rng(seed)
    subjects = df["subject"].unique()
    isis, _ = compute_dprime_curve(df)
    n_isi = len(isis)
    boot_mat = np.zeros((n_boot, n_isi))

    for b in range(n_boot):
        sampled = rng.choice(subjects, size=len(subjects), replace=True)
        _, dp = compute_dprime_curve(df[df["subject"].isin(sampled)])
        boot_mat[b] = dp

    return {
        "isis": isis,
        "mean": boot_mat.mean(axis=0),
        "sem": boot_mat.std(axis=0, ddof=1),
        "ci_low": np.percentile(boot_mat, 2.5, axis=0),
        "ci_high": np.percentile(boot_mat, 97.5, axis=0),
        "boot_matrix": boot_mat,
    }


# ── split-half reliability of d' curve ───────────────────────────────

def split_half_reliability(df, n_splits=1000, seed=0, method="pearson"):
    """
    Split-half reliability of the population d' vs ISI curve.

    Randomly splits subjects into two halves, computes d' curves,
    and correlates them.

    Returns
    -------
    dict with: correlations, mean_r, ci, raw.
    """
    rng = np.random.default_rng(seed)
    df = df[df["isi"] != -1].copy()
    subjects = df["subject"].unique()

    correlations = []
    for _ in range(n_splits):
        rng.shuffle(subjects)
        half1 = subjects[: len(subjects) // 2]
        half2 = subjects[len(subjects) // 2 :]

        isis1, d1 = compute_dprime_for_subjects(df, half1)
        isis2, d2 = compute_dprime_for_subjects(df, half2)
        assert np.all(isis1 == isis2)

        if method == "pearson":
            r = np.corrcoef(d1, d2)[0, 1]
        else:
            r = spearmanr(d1, d2).correlation
        correlations.append(r)

    correlations = np.array(correlations)
    return {
        "correlations": correlations,
        "mean_r": correlations.mean(),
        "ci": np.percentile(correlations, [2.5, 97.5]),
        "raw": correlations,
    }


# ── d' curve power analysis ─────────────────────────────────────────

def dprime_curve_power_analysis(
    df, min_n=6, max_n=None, n_repeats=300, seed=0, method="pearson"
):
    """
    How split-half reliability of the d' curve changes with sample size.

    Returns
    -------
    dict with: Ns, r_mean, r_ci_low, r_ci_high, r_mat.
    """
    df = df[df["isi"] != -1].copy()
    rng = np.random.default_rng(seed)
    subjects = np.array(df["subject"].unique())

    if max_n is None:
        max_n = len(subjects)
    Ns = np.arange(min_n, max_n + 1)
    r_mat = np.full((len(Ns), n_repeats), np.nan)

    for idx, N in enumerate(Ns):
        for k in range(n_repeats):
            chosen = rng.choice(subjects, size=N, replace=False)
            rng.shuffle(chosen)
            half1, half2 = chosen[: N // 2], chosen[N // 2 :]

            isis1, d1 = compute_dprime_for_subjects(df, half1)
            isis2, d2 = compute_dprime_for_subjects(df, half2)

            valid = ~(np.isnan(d1) | np.isnan(d2))
            d1v, d2v = d1[valid], d2[valid]
            if len(d1v) < 2 or np.std(d1v) == 0 or np.std(d2v) == 0:
                continue

            if method == "pearson":
                r_mat[idx, k] = np.corrcoef(d1v, d2v)[0, 1]
            else:
                r_mat[idx, k] = spearmanr(d1v, d2v).correlation

    return {
        "Ns": Ns,
        "r_mean": np.nanmean(r_mat, axis=1),
        "r_ci_low": np.nanpercentile(r_mat, 2.5, axis=1),
        "r_ci_high": np.nanpercentile(r_mat, 97.5, axis=1),
        "r_mat": r_mat,
    }


# ── itemwise analysis ────────────────────────────────────────────────

def compute_itemwise_rates(results, kind):
    """
    Extract mean itemwise rates (hits or false_alarms) from a
    compute_itemwise_split_half_reliability() result dict.

    Parameters
    ----------
    results : dict
        Output of compute_itemwise_split_half_reliability.
    kind : str
        'hits' or 'false_alarms'.

    Returns
    -------
    pd.Series indexed by item name.
    """
    mat = results["itemwise_responses"][kind]
    return mat.mean(axis=0).dropna()


def cross_experiment_itemwise_correlation(results_a, results_b, kind):
    """
    Correlate itemwise rates between two experiments, restricted to
    items common to both.

    Returns
    -------
    dict with: r, n_items, rates_a, rates_b, common_items.
    """
    ra = compute_itemwise_rates(results_a, kind)
    rb = compute_itemwise_rates(results_b, kind)
    common = sorted(set(ra.index) & set(rb.index))
    ra, rb = ra.loc[common].values, rb.loc[common].values
    r = float(np.corrcoef(ra, rb)[0, 1])
    return {
        "r": r,
        "n_items": len(common),
        "rates_a": ra,
        "rates_b": rb,
        "common_items": common,
    }


def spearman_brown(r):
    """Spearman-Brown prophecy formula for doubling test length."""
    return (2 * r) / (1 + r)


def noise_ceiling(r_a, r_b):
    """Noise ceiling via geometric mean of Spearman-Brown corrected reliabilities."""
    return np.sqrt(spearman_brown(r_a) * spearman_brown(r_b))


def itemwise_power_analysis(
    results_a,
    results_b,
    kind="hits",
    n_boot=200,
    step=5,
):
    """
    Power curve for cross-experiment itemwise correlation.

    Downsamples participants in both experiments from step..min(N_a, N_b),
    computes itemwise correlation at each sample size.

    Returns
    -------
    dict with: Ns, mean, ci (n x 2 array).
    """
    mat_a = results_a["itemwise_responses"][kind]
    mat_b = results_b["itemwise_responses"][kind]
    N_max = min(mat_a.shape[0], mat_b.shape[0])

    Ns = np.arange(step, N_max + 1, step)
    mean_corr, ci_low, ci_high = [], [], []

    for n in Ns:
        boot_corrs = []
        for _ in range(n_boot):
            rng = np.random.default_rng()
            idx_a = rng.choice(mat_a.shape[0], size=n, replace=False)
            idx_b = rng.choice(mat_b.shape[0], size=n, replace=False)

            ra = mat_a.iloc[idx_a].mean(axis=0)
            rb = mat_b.iloc[idx_b].mean(axis=0)
            common = sorted(set(ra.dropna().index) & set(rb.dropna().index))
            if len(common) < 2:
                continue
            boot_corrs.append(np.corrcoef(ra[common], rb[common])[0, 1])

        boot_corrs = np.array(boot_corrs)
        mean_corr.append(np.nanmean(boot_corrs))
        ci_low.append(np.nanpercentile(boot_corrs, 2.5))
        ci_high.append(np.nanpercentile(boot_corrs, 97.5))

    return {
        "Ns": Ns,
        "mean": np.array(mean_corr),
        "ci": np.column_stack([ci_low, ci_high]),
    }


# ── response-timing diagnostics ─────────────────────────────────────

def inter_response_times(exps):
    """Compute all inter-response time intervals (ms) across experiments."""
    all_irts = []
    for exp in exps:
        t = np.array(exp["time_elapsed"], dtype=float)
        all_irts.extend(np.diff(t).tolist())
    return np.array(all_irts)


def yes_rate_from_exps(exps):
    """Mean participant-level yes-rate (response >= 1)."""
    rates = []
    for exp in exps:
        resp = pd.to_numeric(exp.response, errors="coerce")
        rates.append(np.mean(resp >= 1))
    return float(np.mean(rates))


# ── stimulus frequency diagnostics ───────────────────────────────────

def stimulus_frequency_by_isi(exps, isis=None):
    """
    Count how often each stimulus appears as a repeat at each ISI.

    Returns
    -------
    dict[int, dict[str, int]]
        ISI -> {stimulus_name: count}.
    """
    import re
    from collections import defaultdict

    if isis is None:
        isis = [0, 1, 2, 4, 8, 16, 32, 64]

    freq = {isi: defaultdict(int) for isi in isis}
    for exp in exps:
        for i in exp.index:
            row_isi = exp.loc[i].isi
            if row_isi in freq:
                stim = exp.loc[i].stimulus.split("/")[-1]
                freq[row_isi][stim] += 1
    return {k: dict(v) for k, v in freq.items()}


# ── p-value formatting ───────────────────────────────────────────────

def p_to_stars(p):
    """Convert p-value to significance stars."""
    if p < 1e-4:
        return f"**** p={p:.2e}"
    if p < 1e-3:
        return f"*** p={p:.4f}"
    if p < 0.01:
        return f"** p={p:.4f}"
    if p < 0.05:
        return f"* p={p:.4f}"
    return f"n.s. p={p:.4f}"
