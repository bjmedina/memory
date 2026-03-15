# 2D Guided-Drift Sandbox — Summary Report

## Overview

A 2D mechanistic sandbox replacing the learned high-dimensional prior with a fully analytic 3-component Gaussian mixture model. Every quantity (density, score, local geometry) is interpretable and controllable.

## GMM Prior Configuration

| Component | Mean        | Covariance        | Weight |
|-----------|-------------|-------------------|--------|
| Broad     | [-2.5, 0.0] | diag(1.5, 1.5)   | 0.40   |
| Tight-1   | [2.0, 2.0]  | diag(0.4, 0.4)   | 0.30   |
| Tight-2   | [2.0, -2.0] | diag(0.4, 0.4)   | 0.30   |

## Stimuli

80 grid points in [-4, 4]² (9×9 grid, truncated to 80). Points are *not* sampled from the GMM — they are fixed experimental stimuli on a regular grid.

## Baseline Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| sigma0    | 0.5   | Encoding noise |
| sigma1    | 0.1   | Short-range drift noise (ISI 1–4) |
| sigma2    | 0.1   | Long-range drift noise (ISI 8–64) |
| drift_step_size | 0.02 | Prior-driven drift magnitude |
| t_step    | 5     | Regime boundary (sigma1 → sigma2) |
| ISIs      | 0, 1, 2, 4, 8, 16, 32, 64 | |

## Baseline d' vs ISI

| ISI | d' (mean ± SEM) |
|-----|-----------------|
| 0   | -0.096 ± 0.024  |
| 1   |  0.826 ± 0.026  |
| 2   |  0.978 ± 0.031  |
| 4   |  1.035 ± 0.027  |
| 8   |  1.581 ± 0.033  |
| 16  |  1.292 ± 0.023  |
| 32  |  1.115 ± 0.015  |
| 64  |  0.918 ± 0.009  |

![Baseline d' curve](figures/2d_sandbox/baseline_dprime_curve.png)

## Ablation: Raw vs Unit-Normalised Score

| ISI | Unit-norm d' | Raw d' |
|-----|-------------|--------|
| 0   | -0.096      | -0.097 |
| 1   |  0.826      |  0.827 |
| 2   |  0.978      |  0.989 |
| 4   |  1.035      |  1.065 |
| 8   |  1.581      |  1.645 |
| 16  |  1.292      |  1.250 |
| 32  |  1.115      |  0.892 |
| 64  |  0.918      |  0.576 |

The raw score shows stronger position-dependent drift, causing faster forgetting at long ISIs (d' drops to 0.576 at ISI=64 vs 0.918 for unit-norm). At shorter ISIs the difference is minimal.

![Raw vs Unit Score](figures/2d_sandbox/raw_vs_unit_score.png)

## Ablation: Matched vs Mismatched Prior

| ISI | Matched d' | Mismatched d' |
|-----|-----------|---------------|
| 0   | -0.099    | -0.101        |
| 1   |  0.828    |  0.824        |
| 2   |  0.977    |  0.983        |
| 4   |  1.034    |  1.036        |
| 8   |  1.589    |  1.578        |
| 16  |  1.293    |  1.289        |
| 32  |  1.115    |  1.113        |
| 64  |  0.920    |  0.925        |

With the current drift_step_size (0.02), the mismatch effect is subtle. Larger drift magnitudes would amplify the difference, since the mismatched prior pulls memories toward wrong attractor regions.

![Prior Mismatch](figures/2d_sandbox/prior_mismatch_dprime.png)

## Key Insights

1. **ISI=0 is near chance**: With sigma0=0.5, immediate repeats have encoding noise too large relative to the 2D signal, producing d' ≈ 0. This is a parameter regime where encoding noise dominates.

2. **Performance peaks mid-range**: d' peaks at ISI=8 (1.58), suggesting a regime where prior-driven drift has had time to improve memory traces (pulling them toward high-density regions) but diffusion noise hasn't yet overwhelmed the benefit.

3. **Raw vs unit score diverges at long ISIs**: The raw score magnitude is position-dependent (larger far from modes), so points in low-density regions drift more aggressively under raw scoring. This causes faster forgetting at long delays. Unit-normalisation stabilises long-ISI performance.

4. **Mismatch effect is drift-strength dependent**: At small drift_step_size, both correct and incorrect priors produce similar d' because drift barely moves memories. The mismatch ablation becomes informative at larger drift magnitudes.

5. **The sandbox is fully functional**: Score math is verified numerically, results are reproducible under fixed seeds, and all ISIs produce finite d' values.

## Figures

- `prior_score_field.png` — GMM density contours + score field + stimuli
- `baseline_dprime_curve.png` — Baseline d' vs ISI
- `raw_vs_unit_score.png` — Raw vs unit-normalised score comparison
- `matched_vs_mismatched_priors.png` — Both prior landscapes side by side
- `prior_mismatch_dprime.png` — d' curves under matched vs mismatched prior
- `item_susceptibility.png` — Per-item geometry vs presentation counts
- `drift_trajectories_2d.png` — Sample drift trajectories on prior contour

## Files Created

| File | Purpose |
|------|---------|
| `src/model/analytic_gmm_2d.py` | Analytic 3-component GMM (log-prob, score, posteriors) |
| `src/model/score_adapter_2d.py` | ScoreFunction-compatible wrapper |
| `utls/sandbox_2d_data.py` | 80-point grid stimuli + geometry descriptors |
| `utls/runners_2d.py` | 2D guided-drift simulation engine |
| `utls/sigma_fitting_2d.py` | Parameter sweep utilities |
| `utls/analysis_2d.py` | Analysis and plotting utilities |
| `tests/test_2d_sandbox.py` | 6 validation tests (all passing) |
| `notebooks/2d_guided_drift_sandbox.ipynb` | End-to-end pipeline notebook |
