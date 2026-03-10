"""
Diagnostic utilities for verifying prior-driven drift behaviour.

The core idea: if drift follows ∇_x log p(x), then iterating

    x ← x + step_size * score(x)

from a noisy starting point should:
  1. Decrease distance to the nearest clean data point.
  2. Decrease the raw score norm (flatter near the mode).
  3. Increase the proxy log-likelihood (negative energy).

Usage (in a notebook)::

    from utls.drift_diagnostics import drift_trajectory, plot_drift_diagnostic

    traj = drift_trajectory(
        score_model=score_fn,      # ScoreFunction instance
        x_start=X[42] + 0.3 * torch.randn_like(X[42]),
        step_size=0.001,
        n_steps=200,
        X_clean=X,                 # optional: dataset for distance tracking
    )
    plot_drift_diagnostic(traj)
"""

import numpy as np
import torch
import matplotlib.pyplot as plt


def drift_trajectory(
    score_model,
    x_start,
    step_size,
    n_steps,
    X_clean=None,
    use_unit_norm=True,
):
    """
    Run gradient-ascent drift from *x_start* and record diagnostics.

    Parameters
    ----------
    score_model : ScoreFunction
        Must expose ``.forward(x)`` (unit-norm) and ``.forward_raw(x)``.
    x_start : Tensor [D] or [1, D]
        Starting point (e.g. a clean encoding + noise).
    step_size : float
        Drift step size per iteration.
    n_steps : int
        Number of drift iterations.
    X_clean : Tensor [N, D] or None
        If provided, tracks min-distance to any row of *X_clean*.
    use_unit_norm : bool
        If True (default), drift uses the unit-norm score (matching the
        runner).  If False, uses the raw score.

    Returns
    -------
    dict
        ``x_trajectory``   : list of [D] tensors (length n_steps+1)
        ``raw_score_norms`` : [n_steps] float array
        ``unit_score_norms``: [n_steps] float array  (should be ≈1)
        ``dist_to_clean``  : [n_steps+1] float array or None
        ``step_sizes_actual``: [n_steps] float — actual L2 displacement per step
    """
    x = x_start.detach().clone().float()
    if x.dim() == 2 and x.shape[0] == 1:
        x = x.squeeze(0)

    device = x.device
    if X_clean is not None:
        X_clean = X_clean.to(device).float()

    trajectory = [x.clone()]
    raw_norms = []
    unit_norms = []
    displacements = []
    dists = []

    if X_clean is not None:
        dists.append(_min_dist(x, X_clean))

    for _ in range(n_steps):
        # raw score (for diagnostics)
        raw = score_model.forward_raw(x.unsqueeze(0))      # [1,1,1,D]
        raw_flat = raw.reshape(-1)
        raw_norms.append(float(raw_flat.norm().cpu()))

        # unit-norm score
        unit = score_model.forward(x.unsqueeze(0))          # [1,1,1,D]
        unit_flat = unit.reshape(-1)
        unit_norms.append(float(unit_flat.norm().cpu()))

        # drift step
        if use_unit_norm:
            dx = step_size * unit_flat
        else:
            dx = step_size * raw_flat

        x = x + dx.to(device)
        displacements.append(float(dx.norm().cpu()))

        trajectory.append(x.clone())
        if X_clean is not None:
            dists.append(_min_dist(x, X_clean))

    return {
        "x_trajectory": trajectory,
        "raw_score_norms": np.array(raw_norms),
        "unit_score_norms": np.array(unit_norms),
        "dist_to_clean": np.array(dists) if dists else None,
        "step_sizes_actual": np.array(displacements),
        "step_size": step_size,
        "n_steps": n_steps,
    }


def _min_dist(x, X_clean):
    """L2 distance from x [D] to nearest row of X_clean [N, D]."""
    return float((X_clean - x.unsqueeze(0)).norm(dim=1).min().cpu())


def plot_drift_diagnostic(traj, title=None):
    """
    Plot drift trajectory diagnostics.

    Panels:
      1. Raw score norm over steps (should decrease toward mode).
      2. Distance to nearest clean point (should decrease).
      3. Actual step displacement (should be roughly constant for unit-norm).
    """
    steps = np.arange(traj["n_steps"])
    n_panels = 2 + (traj["dist_to_clean"] is not None)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    # Panel 1: raw score norm
    ax = axes[0]
    ax.plot(steps, traj["raw_score_norms"], linewidth=1.2)
    ax.set_xlabel("Drift step")
    ax.set_ylabel("||∇ log p(x)||")
    ax.set_title("Raw score norm")
    ax.grid(alpha=0.25)

    # Panel 2: distance to clean
    idx = 1
    if traj["dist_to_clean"] is not None:
        ax = axes[idx]
        ax.plot(np.arange(traj["n_steps"] + 1), traj["dist_to_clean"], linewidth=1.2)
        ax.set_xlabel("Drift step")
        ax.set_ylabel("L2 dist to nearest clean")
        ax.set_title("Distance to data manifold")
        ax.grid(alpha=0.25)
        idx += 1

    # Panel 3: displacement per step
    ax = axes[idx]
    ax.plot(steps, traj["step_sizes_actual"], linewidth=1.2)
    ax.set_xlabel("Drift step")
    ax.set_ylabel("||Δx||")
    ax.set_title(f"Step displacement (step_size={traj['step_size']:.4f})")
    ax.grid(alpha=0.25)

    if title:
        fig.suptitle(title, y=1.03, fontsize=13)
    fig.tight_layout()
    plt.show()
    return fig


def drift_diagnostic_batch(
    score_model,
    X_clean,
    n_samples=10,
    noise_std=0.3,
    step_size=0.001,
    n_steps=200,
    seed=42,
):
    """
    Run drift trajectories from multiple noisy samples and plot aggregate stats.

    Picks *n_samples* random clean points, adds Gaussian noise, drifts
    each back, and plots the mean ± std of raw score norm and distance
    to nearest clean point.

    Returns
    -------
    list[dict]
        Per-sample trajectory dicts.
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X_clean), size=n_samples, replace=False)

    trajs = []
    for idx in indices:
        x_clean = X_clean[idx]
        noise = torch.randn_like(x_clean) * noise_std
        traj = drift_trajectory(
            score_model=score_model,
            x_start=x_clean + noise,
            step_size=step_size,
            n_steps=n_steps,
            X_clean=X_clean,
        )
        trajs.append(traj)

    # Aggregate and plot
    steps = np.arange(n_steps)
    all_norms = np.stack([t["raw_score_norms"] for t in trajs])
    all_dists = np.stack([t["dist_to_clean"] for t in trajs])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    mean_n, std_n = all_norms.mean(0), all_norms.std(0)
    ax.plot(steps, mean_n, linewidth=1.5)
    ax.fill_between(steps, mean_n - std_n, mean_n + std_n, alpha=0.2)
    ax.set_xlabel("Drift step")
    ax.set_ylabel("||∇ log p(x)||")
    ax.set_title(f"Raw score norm (n={n_samples})")
    ax.grid(alpha=0.25)

    ax = axes[1]
    step_ax = np.arange(n_steps + 1)
    mean_d, std_d = all_dists.mean(0), all_dists.std(0)
    ax.plot(step_ax, mean_d, linewidth=1.5)
    ax.fill_between(step_ax, mean_d - std_d, mean_d + std_d, alpha=0.2)
    ax.set_xlabel("Drift step")
    ax.set_ylabel("L2 dist to nearest clean")
    ax.set_title(f"Distance to data (n={n_samples}, noise_std={noise_std})")
    ax.grid(alpha=0.25)

    fig.suptitle(
        f"Drift diagnostic: step_size={step_size}, n_steps={n_steps}",
        y=1.03, fontsize=13,
    )
    fig.tight_layout()
    plt.show()

    return trajs


def _batch_drift_loglik_trajectory(
    score_model,
    x_batch,
    step_size,
    n_steps,
    use_unit_norm=True,
    batch_size=64,
):
    """
    Drift a batch of points and track cumulative Δlog p at each step.

    Since score(x) = ∇_x log p(x), the change in log p from one step is:

        Δlog p ≈ score(x) · Δx = step_size * ||score(x)||²   (raw)
               ≈ step_size * score_raw · score_unit            (unit-norm drift)

    We accumulate these to get relative log p(x_t) - log p(x_0).

    Parameters
    ----------
    score_model : ScoreFunction
    x_batch : Tensor [N, D]
    step_size : float
    n_steps : int
    use_unit_norm : bool
    batch_size : int
        Process points in chunks to avoid OOM.

    Returns
    -------
    delta_logp : ndarray [N, n_steps+1]
        Cumulative Δlog p relative to t=0 (first column is all zeros).
    """
    N, D = x_batch.shape
    device = x_batch.device
    xs = x_batch.detach().clone().float()

    # delta_logp[:, 0] = 0  (reference)
    delta_logp = np.zeros((N, n_steps + 1))

    for step in range(n_steps):
        all_dlp = []
        all_new_xs = []

        for start in range(0, N, batch_size):
            chunk = xs[start : start + batch_size]
            B = chunk.shape[0]

            raw = score_model.forward_raw(chunk).reshape(B, -1)
            unit = score_model.forward(chunk).reshape(B, -1)

            if use_unit_norm:
                dx = step_size * unit
            else:
                dx = step_size * raw

            # Δlog p ≈ raw_score · dx  (first-order Taylor of log p)
            dlp = (raw * dx).sum(dim=1).cpu().numpy()
            all_dlp.append(dlp)
            all_new_xs.append(chunk + dx)

        xs = torch.cat(all_new_xs, dim=0)
        delta_logp[:, step + 1] = delta_logp[:, step] + np.concatenate(all_dlp)

    return delta_logp


def plot_loglik_histograms(
    score_model,
    X_clean,
    n_samples=1000,
    noise_std=0.3,
    step_size=0.001,
    n_steps=200,
    n_snapshots=3,
    seed=42,
    use_unit_norm=True,
    batch_size=64,
):
    """
    Histogram "movie" of relative log p(x) at several time-points.

    Takes *n_samples* clean points, adds noise, drifts them, and shows
    how the distribution of Δlog p evolves.  By default shows 3 panels:
    t=0, t=T/2, t=T.

    Parameters
    ----------
    score_model : ScoreFunction
    X_clean : Tensor [N_total, D]
    n_samples : int
        Points to track (default 1000).
    noise_std : float
        Gaussian noise added to clean points.
    step_size : float
    n_steps : int
    n_snapshots : int
        Number of histogram panels (evenly spaced from 0 to n_steps).
    seed : int
    use_unit_norm : bool
    batch_size : int

    Returns
    -------
    fig : Figure
    delta_logp : ndarray [n_samples, n_steps+1]
    """
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X_clean), size=n_samples, replace=False)
    x_start = X_clean[indices].clone()
    noise = torch.randn_like(x_start) * noise_std
    x_start = x_start + noise

    delta_logp = _batch_drift_loglik_trajectory(
        score_model=score_model,
        x_batch=x_start,
        step_size=step_size,
        n_steps=n_steps,
        use_unit_norm=use_unit_norm,
        batch_size=batch_size,
    )

    # Pick snapshot steps (evenly spaced including endpoints)
    snap_steps = np.linspace(0, n_steps, n_snapshots, dtype=int)

    fig, axes = plt.subplots(1, n_snapshots, figsize=(5 * n_snapshots, 4),
                             sharey=True)
    if n_snapshots == 1:
        axes = [axes]

    # Shared x-range for comparability
    vmin = delta_logp[:, snap_steps].min()
    vmax = delta_logp[:, snap_steps].max()
    bins = np.linspace(vmin - 0.05 * abs(vmax - vmin),
                       vmax + 0.05 * abs(vmax - vmin), 50)

    for ax, t in zip(axes, snap_steps):
        vals = delta_logp[:, t]
        ax.hist(vals, bins=bins, alpha=0.75, edgecolor="black", linewidth=0.5)
        ax.axvline(np.median(vals), color="red", ls="--", alpha=0.7,
                   label=f"median={np.median(vals):.2f}")
        ax.set_xlabel("Δlog p(x) relative to t=0")
        ax.set_title(f"Step {t}")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Count")
    fig.suptitle(
        f"Log-likelihood evolution: {n_samples} points, "
        f"noise_std={noise_std}, step={step_size}",
        y=1.03, fontsize=13,
    )
    fig.tight_layout()
    plt.show()

    return fig, delta_logp
