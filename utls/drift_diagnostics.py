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
    x_source=None,
    use_unit_norm=True,
    knn_k=5,
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
        If provided, tracks distance metrics to *X_clean*.
    x_source : Tensor [D] or None
        The original clean point before corruption.  If provided, tracks
        cosine distance back to the source stimulus.
    use_unit_norm : bool
        If True (default), drift uses the unit-norm score (matching the
        runner).  If False, uses the raw score.
    knn_k : int
        Number of neighbours for median k-NN distance (default 5).

    Returns
    -------
    dict
        ``x_trajectory``      : list of [D] tensors (length n_steps+1)
        ``raw_score_norms``    : [n_steps] float array
        ``unit_score_norms``   : [n_steps] float array  (should be ≈1)
        ``dist_to_clean``      : [n_steps+1] float array — L2 to nearest
        ``median_knn_dist``    : [n_steps+1] float array — median L2 to k-NN
        ``median_knn_cosine_dist``: [n_steps+1] float array — median cosine dist to k-NN
        ``cosine_dist_source`` : [n_steps+1] float array or None
        ``step_sizes_actual``  : [n_steps] float — actual L2 displacement per step
    """
    x = x_start.detach().clone().float()
    if x.dim() == 2 and x.shape[0] == 1:
        x = x.squeeze(0)

    device = x.device
    if X_clean is not None:
        X_clean = X_clean.to(device).float()
    if x_source is not None:
        x_source = x_source.to(device).float()

    trajectory = [x.clone()]
    raw_norms = []
    unit_norms = []
    displacements = []
    dists = []
    knn_dists = []
    cos_dists_nearest = []
    cos_dists_source = []

    if X_clean is not None:
        dists.append(_min_dist(x, X_clean))
        knn_dists.append(_median_knn_dist(x, X_clean, k=knn_k))
        cos_dists_nearest.append(_median_knn_cosine_dist(x, X_clean, k=knn_k))
    if x_source is not None:
        cos_dists_source.append(_cosine_dist_to_source(x, x_source))

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
            knn_dists.append(_median_knn_dist(x, X_clean, k=knn_k))
            cos_dists_nearest.append(_median_knn_cosine_dist(x, X_clean, k=knn_k))
        if x_source is not None:
            cos_dists_source.append(_cosine_dist_to_source(x, x_source))

    return {
        "x_trajectory": trajectory,
        "raw_score_norms": np.array(raw_norms),
        "unit_score_norms": np.array(unit_norms),
        "dist_to_clean": np.array(dists) if dists else None,
        "median_knn_dist": np.array(knn_dists) if knn_dists else None,
        "median_knn_cosine_dist": np.array(cos_dists_nearest) if cos_dists_nearest else None,
        "cosine_dist_source": np.array(cos_dists_source) if cos_dists_source else None,
        "step_sizes_actual": np.array(displacements),
        "step_size": step_size,
        "n_steps": n_steps,
        "knn_k": knn_k,
    }


def _min_dist(x, X_clean):
    """L2 distance from x [D] to nearest row of X_clean [N, D]."""
    return float((X_clean - x.unsqueeze(0)).norm(dim=1).min().cpu())


def _median_knn_dist(x, X_clean, k=5):
    """Median L2 distance from x [D] to its k nearest neighbours in X_clean."""
    dists = (X_clean - x.unsqueeze(0)).norm(dim=1)       # [N]
    topk = dists.topk(k, largest=False).values            # [k]
    return float(topk.median().cpu())


def _median_knn_cosine_dist(x, X_clean, k=5):
    """Median cosine distance (1 - cos_sim) to k nearest neighbours by cosine."""
    x_norm = x / (x.norm() + 1e-12)
    X_norm = X_clean / (X_clean.norm(dim=1, keepdim=True) + 1e-12)
    cos_sims = X_norm @ x_norm                              # [N]
    topk_sims = cos_sims.topk(k, largest=True).values       # [k] highest similarities
    return float((1.0 - topk_sims.median()).cpu())


def _cosine_dist_to_source(x, x_source):
    """Cosine distance (1 - cos_sim) between x [D] and x_source [D]."""
    cos_sim = torch.dot(x, x_source) / (x.norm() * x_source.norm() + 1e-12)
    return float((1.0 - cos_sim).cpu())


def plot_drift_diagnostic(traj, title=None):
    """
    Plot drift trajectory diagnostics.

    Row 1: Raw score norm | L2 to nearest | Step displacement
    Row 2: Median k-NN L2 | Median k-NN cosine | Cosine dist (source)
    """
    steps = np.arange(traj["n_steps"])
    step_ax = np.arange(traj["n_steps"] + 1)
    has_clean = traj["dist_to_clean"] is not None
    has_source = traj.get("cosine_dist_source") is not None

    n_rows = 1 + has_clean
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    # --- Row 1 ---
    ax = axes[0, 0]
    ax.plot(steps, traj["raw_score_norms"], linewidth=1.2)
    ax.set_xlabel("Drift step")
    ax.set_ylabel("||∇ log p(x)||")
    ax.set_title("Raw score norm")
    ax.grid(alpha=0.25)

    if has_clean:
        ax = axes[0, 1]
        ax.plot(step_ax, traj["dist_to_clean"], linewidth=1.2)
        ax.set_xlabel("Drift step")
        ax.set_ylabel("L2 dist to nearest clean")
        ax.set_title("L2 to nearest")
        ax.grid(alpha=0.25)
    else:
        axes[0, 1].axis("off")

    ax = axes[0, 2]
    ax.plot(steps, traj["step_sizes_actual"], linewidth=1.2)
    ax.set_xlabel("Drift step")
    ax.set_ylabel("||Δx||")
    ax.set_title(f"Step displacement (step_size={traj['step_size']:.4f})")
    ax.grid(alpha=0.25)

    # --- Row 2 (distance metrics) ---
    if has_clean:
        knn_k = traj.get("knn_k", 5)

        ax = axes[1, 0]
        ax.plot(step_ax, traj["median_knn_dist"], linewidth=1.2, color="C1")
        ax.set_xlabel("Drift step")
        ax.set_ylabel(f"Median L2 to {knn_k}-NN")
        ax.set_title(f"Median {knn_k}-NN distance")
        ax.grid(alpha=0.25)

        ax = axes[1, 1]
        ax.plot(step_ax, traj["median_knn_cosine_dist"], linewidth=1.2, color="C2")
        ax.set_xlabel("Drift step")
        ax.set_ylabel(f"Median cosine dist to {knn_k}-NN")
        ax.set_title(f"Median {knn_k}-NN cosine dist")
        ax.grid(alpha=0.25)

        ax = axes[1, 2]
        if has_source:
            ax.plot(step_ax, traj["cosine_dist_source"], linewidth=1.2, color="C3")
            ax.set_xlabel("Drift step")
            ax.set_ylabel("Cosine distance")
            ax.set_title("Cosine dist to source")
            ax.grid(alpha=0.25)
        else:
            ax.axis("off")

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
    knn_k=5,
):
    """
    Run drift trajectories from multiple noisy samples and plot aggregate stats.

    Picks *n_samples* random clean points, adds Gaussian noise, drifts
    each back, and plots the mean +/- std of:
      - Raw score norm
      - L2 to nearest clean
      - Median k-NN distance
      - Median k-NN cosine distance
      - Cosine dist to source

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
            x_source=x_clean,
            knn_k=knn_k,
        )
        trajs.append(traj)

    # Aggregate
    steps = np.arange(n_steps)
    step_ax = np.arange(n_steps + 1)

    all_norms = np.stack([t["raw_score_norms"] for t in trajs])
    all_l2 = np.stack([t["dist_to_clean"] for t in trajs])
    all_knn = np.stack([t["median_knn_dist"] for t in trajs])
    all_cos_near = np.stack([t["median_knn_cosine_dist"] for t in trajs])
    all_cos_src = np.stack([t["cosine_dist_source"] for t in trajs])

    def _band(ax, x, arr, **kwargs):
        m, s = arr.mean(0), arr.std(0)
        line, = ax.plot(x, m, linewidth=1.5, **kwargs)
        ax.fill_between(x, m - s, m + s, alpha=0.2, color=line.get_color())

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # Row 1
    ax = axes[0, 0]
    _band(ax, steps, all_norms)
    ax.set_xlabel("Drift step")
    ax.set_ylabel("||∇ log p(x)||")
    ax.set_title(f"Raw score norm (n={n_samples})")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    _band(ax, step_ax, all_l2)
    ax.set_xlabel("Drift step")
    ax.set_ylabel("L2 dist to nearest clean")
    ax.set_title(f"L2 to nearest (n={n_samples})")
    ax.grid(alpha=0.25)

    ax = axes[0, 2]
    _band(ax, step_ax, all_knn, color="C1")
    ax.set_xlabel("Drift step")
    ax.set_ylabel(f"Median L2 to {knn_k}-NN")
    ax.set_title(f"Median {knn_k}-NN distance (n={n_samples})")
    ax.grid(alpha=0.25)

    # Row 2
    ax = axes[1, 0]
    _band(ax, step_ax, all_cos_near, color="C2")
    ax.set_xlabel("Drift step")
    ax.set_ylabel(f"Median cosine dist to {knn_k}-NN")
    ax.set_title(f"Median {knn_k}-NN cosine dist (n={n_samples})")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    _band(ax, step_ax, all_cos_src, color="C3")
    ax.set_xlabel("Drift step")
    ax.set_ylabel("Cosine distance")
    ax.set_title(f"Cosine dist to source (n={n_samples})")
    ax.grid(alpha=0.25)

    axes[1, 2].axis("off")

    fig.suptitle(
        f"Drift diagnostic: step_size={step_size}, n_steps={n_steps}, "
        f"noise_std={noise_std}",
        y=1.02, fontsize=13,
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
