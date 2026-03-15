"""
utls.sigma_fitting_2d — parameter sweep utilities for the 2D sandbox.

Uses the two-parameter noise model:
  - σ₀ : encoding noise (applied once at memory insertion)
  - σ  : diffusive noise (constant per-step noise during Langevin dynamics)

Pure exploration: sweep each parameter over a grid, record the resulting
d' curves.  No target-based fitting — the goal is to understand how
model behavior changes as a function of each parameter.
"""

import numpy as np
import pandas as pd

from utls.runners_2d import run_2d_isi_sweep
from utls.sigma_fitting import make_grid, log_mid


# ── generic 1-D sweep ────────────────────────────────────────────────

def sweep_param(
    score_model,
    X0,
    name_to_idx,
    stimulus_pool,
    param_name,
    param_values,
    fixed_params,
    isi_values=(0, 1, 2, 4, 8, 16, 32, 64),
    n_experiments=20,
    k_stimuli=10,
    n_mc=16,
    seed=42,
    verbose=True,
):
    """Sweep *param_name* over *param_values*, fixing everything else.

    Parameters
    ----------
    param_name : str
        One of ``"sigma0"``, ``"sigma"``, ``"drift_step_size"``.
    param_values : sequence of float
        Values to evaluate.
    fixed_params : dict
        Keys ``sigma0``, ``sigma``, ``drift_step_size``
        (all required except the one being swept).

    Returns
    -------
    pd.DataFrame
        Columns: ``param_value``, ``isi``, ``dprime_mean``,
        ``dprime_sem``, ``auroc``.
    """
    rows = []
    for i, val in enumerate(param_values):
        params = dict(fixed_params)
        params[param_name] = val

        if verbose:
            print(f"  [{i+1}/{len(param_values)}] {param_name}={val:.4g}")

        sweep = run_2d_isi_sweep(
            sigma0=params["sigma0"],
            sigma=params["sigma"],
            drift_step_size=params["drift_step_size"],
            score_model=score_model,
            X0=X0,
            name_to_idx=name_to_idx,
            stimulus_pool=stimulus_pool,
            isi_values=isi_values,
            n_experiments=n_experiments,
            k_stimuli=k_stimuli,
            n_mc=n_mc,
            seed=seed + i * 7,
        )

        for isi, dp_m, dp_s, auc in zip(
            sweep["isi_values"],
            sweep["dprime_mean"],
            sweep["dprime_sem"],
            sweep["auroc"],
        ):
            rows.append({
                "param_value": val,
                "isi": isi,
                "dprime_mean": dp_m,
                "dprime_sem": dp_s,
                "auroc": auc,
            })

    return pd.DataFrame(rows)


# ── sweep with iterative refinement ──────────────────────────────────

def sweep_with_refinement(
    score_model,
    X0,
    name_to_idx,
    stimulus_pool,
    param_name,
    bounds,
    fixed_params,
    n_grid=15,
    n_refine_iters=2,
    isi_values=(0, 1, 2, 4, 8, 16, 32, 64),
    n_mc=16,
    seed=42,
    spacing="log",
    verbose=True,
):
    """1-D grid sweep with zoom-in refinement.

    After each pass, narrows the search range around the region where
    the d' at the smallest non-zero ISI transitions most steeply.

    Returns
    -------
    all_results : pd.DataFrame
        Combined results from all refinement iterations.
    best_value : float
        Parameter value in the final pass with median d' closest to 1.5
        (a moderate discrimination level).
    """
    lo, hi = bounds
    all_dfs = []

    for it in range(1 + n_refine_iters):
        grid = make_grid(lo, hi, n_grid, spacing=spacing)
        if verbose:
            print(f"Refinement iter {it}: {param_name} in [{lo:.4g}, {hi:.4g}]")

        df = sweep_param(
            score_model=score_model,
            X0=X0,
            name_to_idx=name_to_idx,
            stimulus_pool=stimulus_pool,
            param_name=param_name,
            param_values=grid,
            fixed_params=fixed_params,
            isi_values=isi_values,
            n_mc=n_mc,
            seed=seed + it * 1000,
            verbose=verbose,
        )
        df["refine_iter"] = it
        all_dfs.append(df)

        # Zoom in: find the grid point whose median d' (across ISIs) is
        # closest to 1.5 and narrow to its neighbours.
        median_dp = df.groupby("param_value")["dprime_mean"].median()
        target_dp = 1.5
        best_idx = (median_dp - target_dp).abs().idxmin()
        sorted_vals = sorted(median_dp.index)
        pos = sorted_vals.index(best_idx)
        new_lo = sorted_vals[max(0, pos - 1)]
        new_hi = sorted_vals[min(len(sorted_vals) - 1, pos + 1)]
        lo, hi = float(new_lo), float(new_hi)

    all_results = pd.concat(all_dfs, ignore_index=True)
    return all_results, float(best_idx)


# ── staged sweep ──────────────────────────────────────────────────────

def staged_sweep_2d(
    score_model,
    X0,
    name_to_idx,
    stimulus_pool,
    sigma0_bounds=(0.01, 5.0),
    sigma_bounds=(0.001, 2.0),
    drift_bounds=(0.0, 0.5),
    n_grid=15,
    n_refine_iters=2,
    n_mc=16,
    seed=42,
    verbose=True,
):
    """Staged parameter sweep: σ₀ → σ → drift_step_size.

    Each stage sweeps one parameter, auto-selects a moderate value,
    then fixes it for the next stage.

    Returns
    -------
    dict
        ``sweep_sigma0`` : DataFrame
        ``sweep_sigma``  : DataFrame
        ``sweep_drift``  : DataFrame
        ``selected``     : dict of auto-selected parameter values
    """
    selected = {}

    # Stage A: sweep sigma0 at ISI=0
    if verbose:
        print("=" * 60)
        print("Stage A: sweep sigma0 (ISI=0)")
        print("=" * 60)
    df_s0, best_s0 = sweep_with_refinement(
        score_model=score_model,
        X0=X0,
        name_to_idx=name_to_idx,
        stimulus_pool=stimulus_pool,
        param_name="sigma0",
        bounds=sigma0_bounds,
        fixed_params={"sigma0": 0.1, "sigma": 0.0,
                       "drift_step_size": 0.0},
        n_grid=n_grid,
        n_refine_iters=n_refine_iters,
        isi_values=(0,),
        n_mc=n_mc,
        seed=seed,
        verbose=verbose,
    )
    selected["sigma0"] = best_s0
    if verbose:
        print(f"→ Selected sigma0 = {best_s0:.4g}\n")

    # Stage B: sweep sigma (diffusive noise) at ISI 1-4
    if verbose:
        print("=" * 60)
        print("Stage B: sweep sigma (ISI 1,2,4)")
        print("=" * 60)
    df_s, best_s = sweep_with_refinement(
        score_model=score_model,
        X0=X0,
        name_to_idx=name_to_idx,
        stimulus_pool=stimulus_pool,
        param_name="sigma",
        bounds=sigma_bounds,
        fixed_params={"sigma0": best_s0, "sigma": 0.0,
                       "drift_step_size": 0.0},
        n_grid=n_grid,
        n_refine_iters=n_refine_iters,
        isi_values=(1, 2, 4),
        n_mc=n_mc,
        seed=seed + 100,
        verbose=verbose,
    )
    selected["sigma"] = best_s
    if verbose:
        print(f"→ Selected sigma = {best_s:.4g}\n")

    # Stage C: sweep drift_step_size at ISI 8-64
    if verbose:
        print("=" * 60)
        print("Stage C: sweep drift_step_size (ISI 8,16,32,64)")
        print("=" * 60)
    df_drift, best_drift = sweep_with_refinement(
        score_model=score_model,
        X0=X0,
        name_to_idx=name_to_idx,
        stimulus_pool=stimulus_pool,
        param_name="drift_step_size",
        bounds=drift_bounds,
        fixed_params={"sigma0": best_s0, "sigma": best_s,
                       "drift_step_size": 0.0},
        n_grid=n_grid,
        n_refine_iters=n_refine_iters,
        isi_values=(8, 16, 32, 64),
        n_mc=n_mc,
        seed=seed + 200,
        spacing="linear",
        verbose=verbose,
    )
    selected["drift_step_size"] = best_drift
    if verbose:
        print(f"→ Selected drift_step_size = {best_drift:.4g}\n")

    return {
        "sweep_sigma0": df_s0,
        "sweep_sigma": df_s,
        "sweep_drift": df_drift,
        "selected": selected,
    }
