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

## Noise Model (Two-Parameter)

Following the paper's formulation (Equations 1–3):

| Parameter | Symbol | Value | Description |
|-----------|--------|-------|-------------|
| Encoding noise | σ₀ | 0.5 | Applied once at memory insertion (Eq. 1) |
| Diffusive noise | σ | 0.1 | Constant per-step noise during Langevin dynamics (Eq. 2) |
| Drift step size | η | 0.02 | Prior-driven drift magnitude (Eq. 2) |

**Encoding** (Eq. 1): m_i = x_i + σ₀ · s ⊙ ε

**Per-trial update** (Eq. 2): m_j ← m_j + η · ∇ log π(m_j) + σ · s̃ ⊙ ε_j

## Key Insights

1. **ISI=0 is near chance**: With σ₀=0.5, immediate repeats have encoding noise too large relative to the 2D signal, producing d' ≈ 0. This is a parameter regime where encoding noise dominates.

2. **Performance peaks mid-range**: d' peaks around ISI=8, suggesting a regime where prior-driven drift has had time to improve memory traces (pulling them toward high-density regions) but diffusion noise hasn't yet overwhelmed the benefit.

3. **Raw vs unit score diverges at long ISIs**: The raw score magnitude is position-dependent (larger far from modes), so points in low-density regions drift more aggressively under raw scoring. This causes faster forgetting at long delays. Unit-normalisation stabilises long-ISI performance.

4. **Mismatch effect is drift-strength dependent**: At small η, both correct and incorrect priors produce similar d' because drift barely moves memories. The mismatch ablation becomes informative at larger drift magnitudes.

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
