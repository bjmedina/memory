"""
Analytic 2D Gaussian Mixture Model for the mechanistic sandbox.

Provides closed-form log-probability, score (∇ log p), component
posteriors, and posterior entropy for a K-component GMM in ℝ².
All heavy computation uses PyTorch tensors so the outputs plug
directly into the guided-drift runner.
"""

import torch
import numpy as np
from typing import Optional, Sequence


# ── default prior configuration ──────────────────────────────────────

DEFAULT_MEANS = [
    [-2.5, 0.0],   # broad component (centre-left)
    [2.0,  2.0],   # tight component (upper-right)
    [2.0, -2.0],   # tight component (lower-right)
]

DEFAULT_COVARIANCES = [
    [[1.5, 0.0], [0.0, 1.5]],   # broad
    [[0.4, 0.0], [0.0, 0.4]],   # tight
    [[0.4, 0.0], [0.0, 0.4]],   # tight
]

DEFAULT_WEIGHTS = [0.4, 0.3, 0.3]


# ── main class ────────────────────────────────────────────────────────

class AnalyticGMM2D:
    """3-component (or K-component) Gaussian mixture in ℝ².

    Parameters
    ----------
    means : sequence of [2] arrays, length K
    covariances : sequence of [2,2] arrays, length K
    weights : sequence of floats, length K  (must sum to 1)
    """

    def __init__(
        self,
        means: Optional[Sequence] = None,
        covariances: Optional[Sequence] = None,
        weights: Optional[Sequence] = None,
    ):
        means = means if means is not None else DEFAULT_MEANS
        covariances = covariances if covariances is not None else DEFAULT_COVARIANCES
        weights = weights if weights is not None else DEFAULT_WEIGHTS

        self.K = len(means)
        self.D = 2

        # Store as torch tensors (float64 for numerical accuracy)
        self.means = torch.tensor(means, dtype=torch.float64)          # [K, 2]
        self.covs = torch.tensor(covariances, dtype=torch.float64)     # [K, 2, 2]
        self.weights = torch.tensor(weights, dtype=torch.float64)      # [K]

        # Pre-compute inverses and log-normalising constants
        self.prec = torch.linalg.inv(self.covs)                        # [K, 2, 2]
        self.log_dets = torch.logdet(self.covs)                        # [K]
        self.log_norm = -0.5 * (self.D * np.log(2 * np.pi) + self.log_dets)  # [K]

    # ── component log-densities ───────────────────────────────────

    def _component_log_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Log N(x | μ_k, Σ_k) for each component.

        Parameters
        ----------
        x : [B, 2] or [2]

        Returns
        -------
        [B, K] or [K]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        x = x.to(torch.float64)
        diff = x.unsqueeze(1) - self.means.unsqueeze(0)       # [B, K, 2]
        # Mahalanobis: diff @ Σ⁻¹ @ diff
        mahal = torch.einsum("bki,kij,bkj->bk", diff, self.prec, diff)  # [B, K]
        log_probs = self.log_norm.unsqueeze(0) - 0.5 * mahal            # [B, K]

        return log_probs.squeeze(0) if squeeze else log_probs

    # ── mixture log-probability ───────────────────────────────────

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """log p(x) for the full mixture.

        Parameters
        ----------
        x : [B, 2] or [2]

        Returns
        -------
        [B] or scalar
        """
        comp_lp = self._component_log_probs(x)     # [B, K] or [K]
        log_w = torch.log(self.weights)             # [K]
        return torch.logsumexp(comp_lp + log_w, dim=-1)

    def prob(self, x: torch.Tensor) -> torch.Tensor:
        """p(x) = exp(log p(x))."""
        return torch.exp(self.log_prob(x))

    # ── score function ────────────────────────────────────────────

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """∇_x log p(x)  — the score function.

        score(x) = Σ_k  r_k(x) · Σ_k⁻¹ (μ_k − x)

        where r_k(x) = posterior responsibility of component k.

        Parameters
        ----------
        x : [B, 2] or [2]

        Returns
        -------
        Same shape as x.
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        x = x.to(torch.float64)
        r = self.component_posteriors(x)                          # [B, K]
        diff = self.means.unsqueeze(0) - x.unsqueeze(1)          # [B, K, 2]  (μ_k - x)
        # Σ_k⁻¹ (μ_k - x)
        prec_diff = torch.einsum("kij,bkj->bki", self.prec, diff)  # [B, K, 2]
        # weighted sum over components
        s = torch.einsum("bk,bki->bi", r, prec_diff)               # [B, 2]

        return s.squeeze(0) if squeeze else s

    # ── posteriors / responsibilities ─────────────────────────────

    def component_posteriors(self, x: torch.Tensor) -> torch.Tensor:
        """r_k(x) = w_k N(x|μ_k,Σ_k) / p(x).

        Parameters
        ----------
        x : [B, 2] or [2]

        Returns
        -------
        [B, K] or [K]  — posterior responsibilities (sum to 1 over k).
        """
        comp_lp = self._component_log_probs(x)           # [*, K]
        log_w = torch.log(self.weights)                   # [K]
        log_joint = comp_lp + log_w                       # [*, K]
        return torch.softmax(log_joint, dim=-1)

    def posterior_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """H[r(x)] = -Σ_k r_k log r_k.

        Parameters
        ----------
        x : [B, 2] or [2]

        Returns
        -------
        [B] or scalar
        """
        r = self.component_posteriors(x)    # [*, K]
        # Clamp to avoid log(0)
        log_r = torch.log(r.clamp(min=1e-30))
        return -(r * log_r).sum(dim=-1)


# ── convenience constructors ──────────────────────────────────────────

def make_default_gmm() -> AnalyticGMM2D:
    """Return the default 3-component GMM used throughout the sandbox."""
    return AnalyticGMM2D()


def make_mismatched_gmm() -> AnalyticGMM2D:
    """Return a 'wrong' prior for mismatch ablations.

    Shifts all means by (+1.5, +1.0) and inflates covariances by 2×,
    so the prior landscape is systematically wrong about where the
    high-density regions are.
    """
    shifted_means = [
        [-1.0, 1.0],
        [3.5, 3.0],
        [3.5, -1.0],
    ]
    inflated_covs = [
        [[3.0, 0.0], [0.0, 3.0]],
        [[0.8, 0.0], [0.0, 0.8]],
        [[0.8, 0.0], [0.0, 0.8]],
    ]
    return AnalyticGMM2D(
        means=shifted_means,
        covariances=inflated_covs,
        weights=[0.4, 0.3, 0.3],
    )
