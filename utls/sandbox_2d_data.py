"""
Stimulus generation for the 2D guided-drift sandbox.

Creates 80 fixed grid points in [-5, 5]² as experimental stimuli,
plus per-point geometry descriptors (log-density, score norm,
distance to nearest mixture mean, posterior entropy).
"""

import torch
import numpy as np
import pandas as pd
from src.model.analytic_gmm_2d import AnalyticGMM2D


def make_2d_grid_stimuli(n_side=9, lo=-4.0, hi=4.0):
    """Generate grid points in 2D as experimental stimuli.

    Parameters
    ----------
    n_side : int
        Points per side of the grid (n_side² total before truncation).
    lo, hi : float
        Coordinate range (stays inside [-5, 5] support).

    Returns
    -------
    X0 : torch.Tensor [80, 2]
    name_to_idx : dict  str → int
    stimulus_pool : list[str]
    """
    xs = np.linspace(lo, hi, n_side)
    ys = np.linspace(lo, hi, n_side)
    grid = np.array([(x, y) for x in xs for y in ys])  # n_side² × 2

    # Truncate to 80 points (drop last point)
    grid = grid[:80]

    X0 = torch.tensor(grid, dtype=torch.float32)
    stimulus_pool = [f"pt_{i:02d}" for i in range(len(X0))]
    name_to_idx = {name: i for i, name in enumerate(stimulus_pool)}

    return X0, name_to_idx, stimulus_pool


def compute_geometry_descriptors(X0, gmm):
    """Compute per-point geometry descriptors.

    Parameters
    ----------
    X0 : torch.Tensor [N, 2]
    gmm : AnalyticGMM2D

    Returns
    -------
    pd.DataFrame with columns:
        point_id, x, y, log_density, score_norm,
        dist_to_nearest_mean, posterior_entropy
    """
    N = X0.shape[0]
    x64 = X0.to(torch.float64)

    log_density = gmm.log_prob(x64).numpy()
    score_vec = gmm.score(x64)                           # [N, 2]
    score_norm = score_vec.norm(dim=1).numpy()
    post_ent = gmm.posterior_entropy(x64).numpy()

    # Distance to nearest mixture mean
    means = gmm.means  # [K, 2]
    dists_to_means = torch.cdist(x64, means)              # [N, K]
    dist_nearest = dists_to_means.min(dim=1).values.numpy()

    df = pd.DataFrame({
        "point_id": [f"pt_{i:02d}" for i in range(N)],
        "x": X0[:, 0].numpy(),
        "y": X0[:, 1].numpy(),
        "log_density": log_density,
        "score_norm": score_norm,
        "dist_to_nearest_mean": dist_nearest,
        "posterior_entropy": post_ent,
    })
    return df
