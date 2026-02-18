"""
Modular sigma-fitting utilities for the three-regime noise model.

Provides functions to fit sigma0, sigma1, and sigma2 independently
using toy experiments at appropriate ISI ranges, following a sequential
fitting strategy:

  1. Fit sigma0 using ISI-0 experiments
  2. Fix sigma0, fit sigma1 using ISI 1-4 experiments
  3. Fix sigma0 and sigma1, fit sigma2 using ISI 8-64 experiments

Each stage performs a cheap 1-D grid search (with iterative refinement)
rather than an expensive multi-dimensional search.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import roc_auc_score
from tqdm.notebook import trange

from utls.toy_experiments import (
    make_toy_experiment_list,
    make_multi_isi_toy_experiments,
)


# ── helpers ──────────────────────────────────────────────────────────

def log_mid(lo, hi):
    """Geometric mean of *lo* and *hi*."""
    lo, hi = max(lo, 1e-12), max(hi, 1e-12)
    return float(np.exp(0.5 * (np.log(lo) + np.log(hi))))


def make_grid(lo, hi, n, spacing="log"):
    """Return a 1-D grid of *n* values between *lo* and *hi*."""
    lo, hi = float(lo), float(hi)
    if n <= 1 or lo == hi:
        return [lo]
    if spacing == "log":
        lo = max(lo, 1e-12)
        hi = max(hi, lo * 1.0001)
        return list(np.exp(np.linspace(np.log(lo), np.log(hi), n)))
    if spacing == "linear":
        return list(np.linspace(lo, hi, n))
    if spacing == "hybrid":
        n1 = n // 2
        n2 = n - n1
        log_part = np.exp(
            np.linspace(np.log(max(lo, 1e-12)), np.log(hi), n1, endpoint=False)
        )
        lin_part = np.linspace(lo, hi, n2)
        return sorted(set(log_part.tolist() + lin_part.tolist()))
    raise ValueError(f"Unknown spacing: {spacing}")


def auc_to_dprime(auc_val, eps=1e-6):
    """Convert AUC to d' via the equal-variance Gaussian model."""
    auc_val = np.clip(auc_val, eps, 1 - eps)
    return float(np.sqrt(2) * norm.ppf(auc_val))


# ── core evaluation ──────────────────────────────────────────────────

def evaluate_sigma_on_toy_experiments(
    run_experiment_fn,
    sigma_value,
    sigma_name,
    fixed_sigmas,
    noise_mode,
    metric,
    X0,
    name_to_idx,
    experiments_by_isi,
    human_dprimes_by_isi,
    t_step,
    n_mc=32,
    seed=0,
):
    """
    Evaluate one sigma candidate on toy experiments across ISI conditions.

    Parameters
    ----------
    run_experiment_fn : callable
        ``run_experiment_scores`` from ``utls.runners_v2``.
    sigma_value : float
        Value of the sigma being evaluated.
    sigma_name : str
        Which sigma (``"sigma0"``, ``"sigma1"``, or ``"sigma2"``).
    fixed_sigmas : dict
        Fixed sigma values, e.g. ``{"sigma0": 5.0, "sigma2": 0.1}``.
    noise_mode, metric, X0, name_to_idx, t_step :
        Forwarded to *run_experiment_fn*.
    experiments_by_isi : dict[int, list[list]]
        ISI -> list of experiment sequences.
    human_dprimes_by_isi : dict[int, float]
        ISI -> human d' at that ISI.
    n_mc : int
        Monte-Carlo repetitions per experiment batch.
    seed : int
        Base random seed.

    Returns
    -------
    dict
        ``sigma_value``, ``dprime_by_isi``, ``mse_by_isi``, ``mse_mean``,
        ``auc_by_isi``.
    """
    sigmas = dict(fixed_sigmas)
    sigmas[sigma_name] = sigma_value

    dprime_by_isi = {}
    mse_by_isi = {}
    auc_by_isi = {}

    for isi_val, exp_list in experiments_by_isi.items():
        all_hits, all_fas = [], []

        for rep in range(n_mc):
            run_out = run_experiment_fn(
                sigma0=sigmas["sigma0"],
                sigma1=sigmas.get("sigma1", 0.0),
                sigma2=sigmas.get("sigma2", 0.0),
                t_step=t_step,
                rate=0,
                noise_mode=noise_mode,
                metric=metric,
                X0=X0,
                name_to_idx=name_to_idx,
                experiment_list=exp_list,
                debug=False,
                seed=seed * 10_000 + isi_val * 1000 + rep,
            )
            all_hits.extend(run_out["hits"])
            all_fas.extend(run_out["fas"])

        if len(all_hits) == 0 or len(all_fas) == 0:
            dprime_by_isi[isi_val] = np.nan
            mse_by_isi[isi_val] = np.nan
            auc_by_isi[isi_val] = np.nan
            continue

        y_true = np.concatenate(
            [np.ones(len(all_hits)), np.zeros(len(all_fas))]
        )
        scores = np.concatenate([all_hits, all_fas])

        auc = roc_auc_score(y_true, -scores)
        dprime = auc_to_dprime(auc)

        dprime_by_isi[isi_val] = dprime
        auc_by_isi[isi_val] = auc

        if isi_val in human_dprimes_by_isi:
            mse_by_isi[isi_val] = (dprime - human_dprimes_by_isi[isi_val]) ** 2
        else:
            mse_by_isi[isi_val] = np.nan

    finite_mse = [v for v in mse_by_isi.values() if np.isfinite(v)]
    mse_mean = float(np.mean(finite_mse)) if finite_mse else np.nan

    return {
        "sigma_value": sigma_value,
        "sigma_name": sigma_name,
        "dprime_by_isi": dprime_by_isi,
        "mse_by_isi": mse_by_isi,
        "mse_mean": mse_mean,
        "auc_by_isi": auc_by_isi,
    }


# ── 1-D grid search with refinement ─────────────────────────────────

def fit_sigma_1d(
    run_experiment_fn,
    sigma_name,
    sigma_bounds,
    fixed_sigmas,
    noise_mode,
    metric,
    X0,
    name_to_idx,
    experiments_by_isi,
    human_dprimes_by_isi,
    t_step,
    n_grid=15,
    n_mc=32,
    n_refine_iters=2,
    spacing="log",
    seed=0,
    verbose=True,
):
    """
    Fit a single sigma via 1-D grid search with iterative refinement.

    Parameters
    ----------
    run_experiment_fn : callable
        ``run_experiment_scores`` from ``utls.runners_v2``.
    sigma_name : str
        ``"sigma0"``, ``"sigma1"``, or ``"sigma2"``.
    sigma_bounds : tuple[float, float]
        ``(lo, hi)`` search interval.
    fixed_sigmas : dict
        Values of the other (already-fitted or initial) sigmas.
    experiments_by_isi : dict[int, list[list]]
        ISI -> list of toy experiment sequences.
    human_dprimes_by_isi : dict[int, float]
        ISI -> target human d'.
    n_grid : int
        Grid points per refinement iteration.
    n_mc : int
        Monte-Carlo reps per candidate.
    n_refine_iters : int
        How many rounds of narrow-and-re-grid.
    spacing : str
        ``"log"`` | ``"linear"`` | ``"hybrid"`` for the first iteration;
        subsequent iterations always use ``"linear"``.
    seed : int
        Base random seed.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        ``best_sigma``, ``best_mse``, ``best_result``, ``all_results``,
        ``bounds_history``.
    """
    lo, hi = sigma_bounds
    all_results = []
    bounds_history = [(lo, hi)]

    for iteration in range(n_refine_iters):
        sp = spacing if iteration == 0 else "linear"
        grid = make_grid(lo, hi, n_grid, spacing=sp)

        if verbose:
            print(
                f"\n--- {sigma_name} iteration {iteration + 1}/"
                f"{n_refine_iters} ---"
            )
            print(
                f"  Bounds: ({lo:.6f}, {hi:.6f}), "
                f"{len(grid)} candidates"
            )

        iter_results = []
        for i in trange(
            len(grid), desc=f"Fitting {sigma_name} (iter {iteration + 1})"
        ):
            result = evaluate_sigma_on_toy_experiments(
                run_experiment_fn=run_experiment_fn,
                sigma_value=grid[i],
                sigma_name=sigma_name,
                fixed_sigmas=fixed_sigmas,
                noise_mode=noise_mode,
                metric=metric,
                X0=X0,
                name_to_idx=name_to_idx,
                experiments_by_isi=experiments_by_isi,
                human_dprimes_by_isi=human_dprimes_by_isi,
                t_step=t_step,
                n_mc=n_mc,
                seed=seed + iteration * 100_000 + i,
            )
            iter_results.append(result)

        all_results.extend(iter_results)

        # ---- select best & refine bounds ----
        mse_vals = np.array([r["mse_mean"] for r in iter_results])
        sigma_vals = np.array([r["sigma_value"] for r in iter_results])
        best_idx = int(np.argmin(mse_vals))
        best_sigma = float(sigma_vals[best_idx])

        if verbose:
            print(
                f"  Best {sigma_name}: {best_sigma:.6f} "
                f"(MSE: {mse_vals[best_idx]:.6f})"
            )

        # narrow to +-2 neighbours
        n_nbr = 2
        lo_idx = max(0, best_idx - n_nbr)
        hi_idx = min(len(sigma_vals) - 1, best_idx + n_nbr)
        lo = float(sigma_vals[lo_idx])
        hi = float(sigma_vals[hi_idx])
        bounds_history.append((lo, hi))

    # ---- final pick across all iterations ----
    mse_all = np.array([r["mse_mean"] for r in all_results])
    best_overall = int(np.argmin(mse_all))

    return {
        "best_sigma": all_results[best_overall]["sigma_value"],
        "best_mse": all_results[best_overall]["mse_mean"],
        "best_result": all_results[best_overall],
        "all_results": all_results,
        "bounds_history": bounds_history,
    }


# ── diagnostics ──────────────────────────────────────────────────────

def plot_sigma_fit(stage_result, human_dprimes_by_isi=None, title=None):
    """
    Two-panel diagnostic for a single ``fit_sigma_1d`` result.

    Left:  sigma vs MSE (lower is better).
    Right: per-ISI model d' vs sigma, with human d' as horizontal lines.
    """
    results = stage_result["all_results"]
    sigma_name = results[0]["sigma_name"]
    sigma_vals = np.array([r["sigma_value"] for r in results])
    mse_vals = np.array([r["mse_mean"] for r in results])

    order = np.argsort(sigma_vals)
    sigma_sorted = sigma_vals[order]
    mse_sorted = mse_vals[order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # ---- left: MSE ----
    ax = axes[0]
    ax.plot(sigma_sorted, mse_sorted, "o-", markersize=4)
    best = stage_result["best_sigma"]
    ax.axvline(best, ls="--", color="red", alpha=0.6, label=f"best={best:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel(sigma_name)
    ax.set_ylabel("MSE vs human d'")
    ax.set_title(f"{sigma_name}: MSE landscape")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    # ---- right: d' per ISI ----
    ax = axes[1]
    isi_keys = sorted(
        {k for r in results for k in r["dprime_by_isi"]},
    )
    for isi_val in isi_keys:
        dp = np.array(
            [r["dprime_by_isi"].get(isi_val, np.nan) for r in results]
        )
        dp_sorted = dp[order]
        ax.plot(sigma_sorted, dp_sorted, "o-", markersize=3, label=f"ISI={isi_val}")

        if human_dprimes_by_isi and isi_val in human_dprimes_by_isi:
            ax.axhline(
                human_dprimes_by_isi[isi_val],
                ls=":",
                alpha=0.5,
                label=f"human ISI={isi_val}",
            )

    ax.axvline(best, ls="--", color="red", alpha=0.6)
    ax.set_xscale("log")
    ax.set_xlabel(sigma_name)
    ax.set_ylabel("d'")
    ax.set_title(f"{sigma_name}: model d' vs human")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.25)

    if title:
        fig.suptitle(title, y=1.03, fontsize=13)
    fig.tight_layout()
    plt.show()

    return fig


# ── three-stage orchestrator ─────────────────────────────────────────

def three_stage_fit(
    run_experiment_fn,
    param_bounds,
    noise_mode,
    metric,
    X0,
    name_to_idx,
    stimulus_pool,
    human_curve,
    isis,
    t_step,
    # ISI groupings
    isi_sigma0=None,
    isi_sigma1=None,
    isi_sigma2=None,
    # toy experiment settings
    n_experiments_per_isi=20,
    k_stimuli_per_exp=10,
    # grid search settings
    n_grid=15,
    n_mc=32,
    n_refine_iters=2,
    seed=0,
    verbose=True,
    plot=True,
):
    """
    Three-stage sequential fitting of sigma0, sigma1, sigma2.

    Stage A — fit sigma0 on ISI-0 toy experiments (only sigma0 matters).
    Stage B — fix sigma0, fit sigma1 on ISI 1-4 toy experiments.
    Stage C — fix sigma0 & sigma1, fit sigma2 on ISI 8-64 toy experiments.

    Parameters
    ----------
    run_experiment_fn : callable
        ``run_experiment_scores`` from ``utls.runners_v2``.
    param_bounds : dict
        ``{"sigma0": (lo, hi), "sigma1": (lo, hi), "sigma2": (lo, hi)}``.
    noise_mode : str
        ``"three-regime"``.
    metric : str
        Distance metric.
    X0 : Tensor
        Encoded stimuli.
    name_to_idx : dict
        Stimulus name -> index.
    stimulus_pool : list
        Stimulus paths available for toy experiments.
    human_curve : array-like
        Human d' values, one per ISI (same order as *isis*).
    isis : list[int]
        ISI values corresponding to *human_curve* entries.
    t_step : int
        Noise regime boundary.
    isi_sigma0, isi_sigma1, isi_sigma2 : list[int] or None
        Override default ISI groupings.
    n_experiments_per_isi : int
        Toy experiments generated per ISI value.
    k_stimuli_per_exp : int
        Stimuli sampled per toy experiment.
    n_grid, n_mc, n_refine_iters :
        Grid search hyper-parameters.
    seed : int
        Base random seed.
    verbose, plot : bool
        Print / plot diagnostics.

    Returns
    -------
    dict
        ``sigma0``, ``sigma1``, ``sigma2``, ``stage_a``, ``stage_b``,
        ``stage_c``.
    """
    if isi_sigma0 is None:
        isi_sigma0 = [0]
    if isi_sigma1 is None:
        isi_sigma1 = [1, 2, 4]
    if isi_sigma2 is None:
        isi_sigma2 = [8, 16, 32, 64]

    # map ISI -> human d'
    isi_to_human = {}
    for i, isi_val in enumerate(isis):
        if i < len(human_curve):
            isi_to_human[isi_val] = float(human_curve[i])

    sigma1_init = log_mid(*param_bounds["sigma1"])
    sigma2_init = log_mid(*param_bounds["sigma2"])

    # ================================================================
    # STAGE A: fit sigma0
    # ================================================================
    if verbose:
        print("=" * 60)
        print("STAGE A: Fitting sigma0 (ISI = 0)")
        print("=" * 60)

    isi0_exps = {
        0: make_toy_experiment_list(
            stimulus_pool,
            isi=0,
            n_experiments=n_experiments_per_isi,
            k_stimuli=k_stimuli_per_exp,
            seed=seed,
        )
    }

    stage_a = fit_sigma_1d(
        run_experiment_fn=run_experiment_fn,
        sigma_name="sigma0",
        sigma_bounds=param_bounds["sigma0"],
        fixed_sigmas={"sigma1": sigma1_init, "sigma2": sigma2_init},
        noise_mode=noise_mode,
        metric=metric,
        X0=X0,
        name_to_idx=name_to_idx,
        experiments_by_isi=isi0_exps,
        human_dprimes_by_isi={0: isi_to_human.get(0, 0.0)},
        t_step=t_step,
        n_grid=n_grid,
        n_mc=n_mc,
        n_refine_iters=n_refine_iters,
        seed=seed,
        verbose=verbose,
    )

    sigma0_fitted = stage_a["best_sigma"]
    if verbose:
        print(f"\n>>> Stage A result: sigma0 = {sigma0_fitted:.6f}")
    if plot:
        plot_sigma_fit(
            stage_a,
            human_dprimes_by_isi={0: isi_to_human.get(0, 0.0)},
            title="Stage A: sigma0 (ISI = 0)",
        )

    # ================================================================
    # STAGE B: fit sigma1
    # ================================================================
    if verbose:
        print("\n" + "=" * 60)
        print(f"STAGE B: Fitting sigma1 (ISIs {isi_sigma1})")
        print("=" * 60)

    sigma1_exps = make_multi_isi_toy_experiments(
        stimulus_pool,
        isi_values=isi_sigma1,
        n_experiments_per_isi=n_experiments_per_isi,
        k_stimuli=k_stimuli_per_exp,
        seed=seed + 1_000,
    )
    sigma1_human = {
        isi: isi_to_human[isi]
        for isi in isi_sigma1
        if isi in isi_to_human
    }

    stage_b = fit_sigma_1d(
        run_experiment_fn=run_experiment_fn,
        sigma_name="sigma1",
        sigma_bounds=param_bounds["sigma1"],
        fixed_sigmas={"sigma0": sigma0_fitted, "sigma2": sigma2_init},
        noise_mode=noise_mode,
        metric=metric,
        X0=X0,
        name_to_idx=name_to_idx,
        experiments_by_isi=sigma1_exps,
        human_dprimes_by_isi=sigma1_human,
        t_step=t_step,
        n_grid=n_grid,
        n_mc=n_mc,
        n_refine_iters=n_refine_iters,
        seed=seed + 100_000,
        verbose=verbose,
    )

    sigma1_fitted = stage_b["best_sigma"]
    if verbose:
        print(f"\n>>> Stage B result: sigma1 = {sigma1_fitted:.6f}")
    if plot:
        plot_sigma_fit(stage_b, human_dprimes_by_isi=sigma1_human,
                       title=f"Stage B: sigma1 (ISIs {isi_sigma1})")

    # ================================================================
    # STAGE C: fit sigma2
    # ================================================================
    if verbose:
        print("\n" + "=" * 60)
        print(f"STAGE C: Fitting sigma2 (ISIs {isi_sigma2})")
        print("=" * 60)

    sigma2_exps = make_multi_isi_toy_experiments(
        stimulus_pool,
        isi_values=isi_sigma2,
        n_experiments_per_isi=n_experiments_per_isi,
        k_stimuli=k_stimuli_per_exp,
        seed=seed + 2_000,
    )
    sigma2_human = {
        isi: isi_to_human[isi]
        for isi in isi_sigma2
        if isi in isi_to_human
    }

    stage_c = fit_sigma_1d(
        run_experiment_fn=run_experiment_fn,
        sigma_name="sigma2",
        sigma_bounds=param_bounds["sigma2"],
        fixed_sigmas={"sigma0": sigma0_fitted, "sigma1": sigma1_fitted},
        noise_mode=noise_mode,
        metric=metric,
        X0=X0,
        name_to_idx=name_to_idx,
        experiments_by_isi=sigma2_exps,
        human_dprimes_by_isi=sigma2_human,
        t_step=t_step,
        n_grid=n_grid,
        n_mc=n_mc,
        n_refine_iters=n_refine_iters,
        seed=seed + 200_000,
        verbose=verbose,
    )

    sigma2_fitted = stage_c["best_sigma"]
    if verbose:
        print(f"\n>>> Stage C result: sigma2 = {sigma2_fitted:.6f}")
    if plot:
        plot_sigma_fit(stage_c, human_dprimes_by_isi=sigma2_human,
                       title=f"Stage C: sigma2 (ISIs {isi_sigma2})")

    # ================================================================
    # summary
    # ================================================================
    if verbose:
        print("\n" + "=" * 60)
        print("THREE-STAGE FIT COMPLETE")
        print("=" * 60)
        print(f"  sigma0 = {sigma0_fitted:.6f}")
        print(f"  sigma1 = {sigma1_fitted:.6f}")
        print(f"  sigma2 = {sigma2_fitted:.6f}")

    return {
        "sigma0": sigma0_fitted,
        "sigma1": sigma1_fitted,
        "sigma2": sigma2_fitted,
        "stage_a": stage_a,
        "stage_b": stage_b,
        "stage_c": stage_c,
    }
