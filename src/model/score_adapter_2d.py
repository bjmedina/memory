"""
Score adapter wrapping :class:`AnalyticGMM2D` to match the
:class:`ScoreFunction` API expected by ``run_model_core_prior``.

Provides ``forward(x)`` (unit-norm) and ``forward_raw(x)`` with the
same input-shape flexibility as ``ScoreFunction``:
  * [D]         → single point
  * [B, D]      → batch
  * [B, 1, 1, D] → 4-D layout used by the audio prior
"""

import torch
from src.model.analytic_gmm_2d import AnalyticGMM2D


class ScoreAdapter2D:
    """Drop-in replacement for ``ScoreFunction`` backed by an analytic GMM.

    Parameters
    ----------
    gmm : AnalyticGMM2D
        The analytic prior.
    normalize : bool
        If True, ``forward`` returns unit-norm scores (like the default
        ``ScoreFunction``).  ``forward_raw`` always returns raw scores.
    """

    def __init__(self, gmm: AnalyticGMM2D, normalize: bool = True):
        self.gmm = gmm
        self.normalize = normalize

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _unpack(x: torch.Tensor):
        """Normalise input to [B, D] and remember how to reshape back."""
        orig_shape = x.shape
        if x.dim() == 1:
            return x.unsqueeze(0), orig_shape, "1d"
        if x.dim() == 2:
            return x, orig_shape, "2d"
        if x.dim() == 4:
            B = x.shape[0]
            return x.reshape(B, -1), orig_shape, "4d"
        raise ValueError(
            f"Unsupported x shape {tuple(x.shape)}; expected [D], [B,D], or [B,1,1,D]."
        )

    @staticmethod
    def _repack(s: torch.Tensor, orig_shape, mode: str) -> torch.Tensor:
        """Restore the output to the caller's original layout."""
        if mode == "1d":
            return s.squeeze(0)
        if mode == "4d":
            B = orig_shape[0]
            return s.reshape(B, 1, 1, -1)
        return s   # "2d" — already [B, D]

    # ── public API (mirrors ScoreFunction) ────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return (optionally unit-norm) score with same layout as *x*."""
        x2d, orig_shape, mode = self._unpack(x)
        dtype_in = x2d.dtype

        score = self.gmm.score(x2d.to(torch.float64)).to(dtype_in)   # [B, 2]

        if self.normalize:
            norms = score.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
            score = score / norms

        return self._repack(score, orig_shape, mode)

    def forward_raw(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw (unnormalized) score ∇_x log p(x)."""
        x2d, orig_shape, mode = self._unpack(x)
        dtype_in = x2d.dtype

        score = self.gmm.score(x2d.to(torch.float64)).to(dtype_in)   # [B, 2]
        return self._repack(score, orig_shape, mode)
