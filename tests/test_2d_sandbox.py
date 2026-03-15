"""
Validation tests for the 2D guided-drift sandbox.

Run with:  python -m pytest tests/test_2d_sandbox.py -v
"""

import sys
import os
import numpy as np
import torch

# Ensure repo root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model.analytic_gmm_2d import AnalyticGMM2D, make_default_gmm
from src.model.score_adapter_2d import ScoreAdapter2D
from utls.sandbox_2d_data import make_2d_grid_stimuli, compute_geometry_descriptors
from utls.runners_2d import run_model_core_2d, run_2d_isi_sweep
from utls.runners_v2 import ThreeRegimeNoise


# ── 1. Finite-difference score check ─────────────────────────────────

def test_finite_diff_score():
    """Analytic score matches finite-difference ∇log p(x) at random points."""
    gmm = make_default_gmm()
    rng = np.random.default_rng(123)
    eps = 1e-5

    for _ in range(10):
        x = torch.tensor(rng.uniform(-4, 4, size=2), dtype=torch.float64)
        analytic = gmm.score(x).numpy()

        numerical = np.zeros(2)
        for d in range(2):
            x_plus = x.clone()
            x_minus = x.clone()
            x_plus[d] += eps
            x_minus[d] -= eps
            numerical[d] = (
                gmm.log_prob(x_plus).item() - gmm.log_prob(x_minus).item()
            ) / (2 * eps)

        np.testing.assert_allclose(
            analytic, numerical, rtol=1e-4, atol=1e-8,
            err_msg=f"Score mismatch at x={x.numpy()}"
        )


# ── 2. Score adapter shape handling ───────────────────────────────────

def test_score_adapter_shapes():
    """ScoreAdapter2D handles [D], [B,D], [B,1,1,D] correctly."""
    gmm = make_default_gmm()
    adapter = ScoreAdapter2D(gmm, normalize=True)

    # [D] → [D]
    x1d = torch.tensor([1.0, -1.0])
    s1d = adapter.forward(x1d)
    assert s1d.shape == (2,), f"Expected (2,), got {s1d.shape}"
    assert torch.isfinite(s1d).all()

    # [B, D] → [B, D]
    x2d = torch.randn(5, 2)
    s2d = adapter.forward(x2d)
    assert s2d.shape == (5, 2), f"Expected (5, 2), got {s2d.shape}"

    # [B, 1, 1, D] → [B, 1, 1, D]
    x4d = torch.randn(3, 1, 1, 2)
    s4d = adapter.forward(x4d)
    assert s4d.shape == (3, 1, 1, 2), f"Expected (3, 1, 1, 2), got {s4d.shape}"

    # Unit-norm check
    norms = s2d.norm(dim=1)
    np.testing.assert_allclose(
        norms.numpy(), np.ones(5), atol=1e-6,
        err_msg="Normalized scores should have unit norm"
    )

    # Raw scores should NOT be unit norm in general
    adapter_raw = ScoreAdapter2D(gmm, normalize=False)
    s_raw = adapter_raw.forward(x2d)
    raw_norms = s_raw.norm(dim=1).numpy()
    assert not np.allclose(raw_norms, 1.0, atol=0.01), \
        "Raw scores should generally not be unit norm"


# ── 3. Seed reproducibility ──────────────────────────────────────────

def test_reproducibility_with_seeds():
    """Two runs with the same seed produce identical results."""
    gmm = make_default_gmm()
    adapter = ScoreAdapter2D(gmm)
    X0, name_to_idx, pool = make_2d_grid_stimuli()
    schedule = ThreeRegimeNoise(0.5, 0.1, 0.1, 5)

    from utls.toy_experiments import make_toy_experiment_list
    exp = make_toy_experiment_list(pool, isi=2, n_experiments=5,
                                   k_stimuli=8, seed=99)

    out1 = run_model_core_2d(
        sigma0=0.5, X0=X0, name_to_idx=name_to_idx,
        experiment_list=exp, score_model=adapter,
        drift_step_size=0.01, noise_schedule=schedule, seed=42,
    )
    out2 = run_model_core_2d(
        sigma0=0.5, X0=X0, name_to_idx=name_to_idx,
        experiment_list=exp, score_model=adapter,
        drift_step_size=0.01, noise_schedule=schedule, seed=42,
    )

    np.testing.assert_array_equal(out1["hits"], out2["hits"])
    np.testing.assert_array_equal(out1["fas"], out2["fas"])


# ── 4. No NaN d' values ──────────────────────────────────────────────

def test_no_nans():
    """Full sweep produces no NaN d' values for reasonable parameters."""
    gmm = make_default_gmm()
    adapter = ScoreAdapter2D(gmm)
    X0, name_to_idx, pool = make_2d_grid_stimuli()

    sweep = run_2d_isi_sweep(
        sigma0=0.5, sigma1=0.1, sigma2=0.1, drift_step_size=0.01,
        score_model=adapter, X0=X0, name_to_idx=name_to_idx,
        stimulus_pool=pool, t_step=5,
        isi_values=(0, 1, 2, 4), n_experiments=10,
        k_stimuli=8, n_mc=4, seed=42,
    )

    dprimes = np.array(sweep["dprime_mean"])
    assert np.all(np.isfinite(dprimes)), \
        f"Found NaN/Inf d' values: {dprimes}"


# ── 5. Log-prob normalization ─────────────────────────────────────────

def test_log_prob_normalization():
    """GMM integrates to approximately 1 via numerical quadrature."""
    gmm = make_default_gmm()

    # Dense grid over [-8, 8]² (wide enough to capture tails)
    n = 200
    xs = np.linspace(-8, 8, n)
    dx = xs[1] - xs[0]
    Xg, Yg = np.meshgrid(xs, xs)
    pts = torch.tensor(
        np.stack([Xg.ravel(), Yg.ravel()], axis=1), dtype=torch.float64
    )

    prob = gmm.prob(pts).numpy()
    integral = prob.sum() * dx * dx

    np.testing.assert_allclose(
        integral, 1.0, atol=0.01,
        err_msg=f"GMM integral = {integral}, expected ~1.0"
    )


# ── 6. Runner output format ──────────────────────────────────────────

def test_runner_output_format():
    """2D runner output has the expected keys."""
    gmm = make_default_gmm()
    adapter = ScoreAdapter2D(gmm)
    X0, name_to_idx, pool = make_2d_grid_stimuli()
    schedule = ThreeRegimeNoise(0.5, 0.1, 0.1, 5)

    from utls.toy_experiments import make_toy_experiment_list
    exp = make_toy_experiment_list(pool, isi=1, n_experiments=3,
                                   k_stimuli=5, seed=0)

    out = run_model_core_2d(
        sigma0=0.5, X0=X0, name_to_idx=name_to_idx,
        experiment_list=exp, score_model=adapter,
        drift_step_size=0.0, noise_schedule=schedule, seed=0,
    )

    expected_keys = {"hits", "fas", "isi_hit_dists", "fa_by_t",
                     "T_max", "score_type", "stds_over_time", "metric"}
    assert expected_keys.issubset(out.keys()), \
        f"Missing keys: {expected_keys - out.keys()}"
    assert isinstance(out["hits"], np.ndarray)
    assert isinstance(out["fas"], np.ndarray)
    assert isinstance(out["isi_hit_dists"], dict)
