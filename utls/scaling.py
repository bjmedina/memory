"""
Universality / rescaling analysis for forgetting curves.

Tests whether different stimulus categories produce the same-shaped
forgetting curve up to a linear transformation, and fits parametric
models (linear, logarithmic, power-law) to the average curve.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr


# ── linear scaling between curves ────────────────────────────────────

def linear_scale(source_curve, target_curve):
    """
    Fit y = a*x + b to map *source_curve* onto *target_curve*.

    Parameters
    ----------
    source_curve, target_curve : array-like
        Must have the same length.

    Returns
    -------
    dict with: a, b, scaled, r (Pearson correlation of scaled vs target).
    """
    source = np.asarray(source_curve).reshape(-1, 1)
    target = np.asarray(target_curve)
    lr = LinearRegression().fit(source, target)
    a, b = float(lr.coef_[0]), float(lr.intercept_)
    scaled = a * source.ravel() + b
    r, _ = pearsonr(scaled, target)
    return {"a": a, "b": b, "scaled": scaled, "r": float(r)}


# ── model fitting ────────────────────────────────────────────────────

def fit_forgetting_models(isis, dprime, zero_substitute=0.1):
    """
    Fit linear, logarithmic, and power-law models to a forgetting curve.

    Parameters
    ----------
    isis : array
        ISI values (may include 0).
    dprime : array
        d' values at each ISI.
    zero_substitute : float
        Value to substitute for ISI=0 in log/power models.

    Returns
    -------
    dict with keys 'linear', 'log', 'power', each containing:
        pred : predicted d' values
        r2 : R-squared on original scale
        params : model parameters
    """
    isi = np.asarray(isis, dtype=float)
    y = np.asarray(dprime)
    isi_safe = np.where(isi == 0, zero_substitute, isi)

    results = {}

    # Linear: d' = a + b * ISI
    X = isi.reshape(-1, 1)
    lr = LinearRegression().fit(X, y)
    pred = lr.predict(X)
    results["linear"] = {
        "pred": pred,
        "r2": float(lr.score(X, y)),
        "params": {"a": float(lr.intercept_), "b": float(lr.coef_[0])},
    }

    # Logarithmic: d' = a + b * log(ISI + 1)
    X_log = np.log(isi_safe + 1).reshape(-1, 1)
    lr = LinearRegression().fit(X_log, y)
    pred = lr.predict(X_log)
    results["log"] = {
        "pred": pred,
        "r2": float(lr.score(X_log, y)),
        "params": {"a": float(lr.intercept_), "b": float(lr.coef_[0])},
    }

    # Power-law: log(d') = A + B * log(ISI)  =>  d' = exp(A) * ISI^B
    y_pos = np.maximum(y, 1e-4)
    X_pow = np.log(isi_safe).reshape(-1, 1)
    y_pow = np.log(y_pos)
    lr = LinearRegression().fit(X_pow, y_pow)
    pred_log = lr.predict(X_pow)
    pred = np.exp(pred_log)
    ss_res = np.sum((y_pos - pred) ** 2)
    ss_tot = np.sum((y_pos - np.mean(y_pos)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    results["power"] = {
        "pred": pred,
        "r2": float(r2),
        "params": {"A": float(lr.intercept_), "B": float(lr.coef_[0])},
    }

    return results


def fit_piecewise_models(isis, dprime, transfer_point=8, zero_substitute=0.1):
    """
    Fit log and power-law models separately to small-ISI and large-ISI regions.

    Parameters
    ----------
    isis : array
        ISI values.
    dprime : array
        d' values.
    transfer_point : int
        Boundary between small and large regions.

    Returns
    -------
    dict with keys 'small' and 'large', each containing 'log' and 'power' fits.
    """
    isi = np.asarray(isis, dtype=float)
    y = np.asarray(dprime)
    isi_safe = np.where(isi == 0, zero_substitute, isi)

    small_mask = isi < transfer_point + 1
    large_mask = isi >= transfer_point

    results = {}
    for name, mask in [("small", small_mask), ("large", large_mask)]:
        isi_r = isi_safe[mask]
        y_r = y[mask]
        region = {}

        # Log fit
        X_log = np.log(isi_r + 1).reshape(-1, 1)
        lr = LinearRegression().fit(X_log, y_r)
        pred = lr.predict(X_log)
        region["log"] = {
            "pred": pred,
            "r2": float(lr.score(X_log, y_r)),
            "isis": isi[mask],
        }

        # Power fit
        y_pos = np.maximum(y_r, 1e-4)
        X_pow = np.log(isi_r).reshape(-1, 1)
        y_pow = np.log(y_pos)
        lr = LinearRegression().fit(X_pow, y_pow)
        pred_pow = np.exp(lr.predict(X_pow))
        ss_res = np.sum((y_pos - pred_pow) ** 2)
        ss_tot = np.sum((y_pos - np.mean(y_pos)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        region["power"] = {
            "pred": pred_pow,
            "r2": float(r2),
            "isis": isi[mask],
        }

        results[name] = region

    return results


# ── plotting ─────────────────────────────────────────────────────────

def plot_universality(
    grid,
    curves,
    names,
    scaled_curves=None,
    avg_curve=None,
    scale_info=None,
    title_prefix="",
    save_path=None,
):
    """
    Three-panel figure: linear, log-x, and log-log views of forgetting curves.

    Parameters
    ----------
    grid : array
        ISI values.
    curves : list[array]
        Raw d' curves, one per category.
    names : list[str]
        Category names.
    scaled_curves : list[array] or None
        Linearly-scaled curves (same length as curves).
    avg_curve : array or None
        Average curve to overlay.
    scale_info : list[dict] or None
        Output of linear_scale() for each scaled curve.
    save_path : str or None
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    grid_log = grid.astype(float).copy()
    grid_log[grid_log == 0] = 0.1

    colors = ["green", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]

    for panel_idx, ax in enumerate(axes):
        for i, (curve, name) in enumerate(zip(curves, names)):
            c = colors[i % len(colors)]
            if panel_idx == 0:
                x = grid
            else:
                x = grid_log
            y = np.maximum(curve, 1e-2) if panel_idx == 2 else curve

            ax.plot(x, y, "o--", alpha=0.25, color=c)
            if panel_idx == 0:
                ax.plot(x, y, "o--", alpha=0.25, color=c, label=f"{name} (raw)")

            if scaled_curves and i < len(scaled_curves):
                sc = scaled_curves[i]
                y_sc = np.maximum(sc, 1e-2) if panel_idx == 2 else sc
                lbl = None
                if panel_idx == 0 and scale_info:
                    lbl = f"{name} (scaled, r={scale_info[i]['r']:.2f})"
                ax.plot(x, y_sc, "o-", linewidth=2, alpha=0.8, color=c, label=lbl)

        if avg_curve is not None:
            y_avg = np.maximum(avg_curve, 1e-2) if panel_idx == 2 else avg_curve
            x_avg = grid if panel_idx == 0 else grid_log
            ax.plot(x_avg, y_avg, "--", linewidth=2.5, alpha=0.8,
                    color="#c084fc", label="Average (raw)")

        if panel_idx >= 1:
            ax.set_xscale("log")
        if panel_idx == 2:
            ax.set_yscale("log")

        ax.set_xlabel("ISI")
        if panel_idx == 0:
            ax.set_ylabel("d'")
            ax.set_xticks(grid)
        ax.grid(True, ls="--", alpha=0.3)

        titles = ["Linear scale", "Log-X scale", "Log-Log scale"]
        ax.set_title(f"{title_prefix}{titles[panel_idx]}")

    axes[0].legend(fontsize=8)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return fig


def plot_model_fits(
    isis,
    target,
    model_results,
    log_x=True,
    title=None,
    save_path=None,
    ax=None,
):
    """
    Overlay model fit predictions on the average curve.

    Parameters
    ----------
    isis : array
        ISI values.
    target : array
        Observed d' values.
    model_results : dict
        Output of fit_forgetting_models().
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(isis, target, "o-", label="Average Data", linewidth=3)

    styles = {"linear": "--", "log": "--", "power": "--"}
    for name, res in model_results.items():
        ax.plot(isis, res["pred"], styles.get(name, "--"),
                label=f"{name.capitalize()} (R²={res['r2']:.2f})")

    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel("ISI")
    ax.set_ylabel("d'")
    if title:
        ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend()

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()


def plot_piecewise_fits(
    isis,
    target,
    piecewise_results,
    title=None,
    save_path=None,
    ax=None,
):
    """
    Plot piecewise log and power fits on small and large ISI regions.

    Parameters
    ----------
    isis : array
        Full ISI grid.
    target : array
        Full d' curve.
    piecewise_results : dict
        Output of fit_piecewise_models().
    """
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(isis, target, "o-", linewidth=3, label="Average Data", color="#1f77b4")

    for region, style in [("small", "--"), ("large", "-.")]:
        for model, color in [("log", "green"), ("power", "red")]:
            res = piecewise_results[region][model]
            label = f"{model.capitalize()} ({region}, R²={res['r2']:.2f})"
            ax.plot(res["isis"], res["pred"], style, linewidth=2, color=color,
                    alpha=0.7 if region == "large" else 1.0, label=label)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("ISI")
    ax.set_ylabel("d'")
    if title:
        ax.set_title(title)
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(fontsize=8)

    if own_fig:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()
