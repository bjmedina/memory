"""
Decaying noise schedules for the hybrid prior-drift model.

All schedules follow:  σ(age) = floor + Δ · f(age, rate)

where `floor` (C) sets the asymptotic noise level and `Δ` controls
the transient excess noise at short ages.

Each schedule accepts scalar or torch.Tensor ages for vectorised
evaluation inside the simulation loop.
"""

import torch
import numpy as np


class DecaySchedule:
    """Base class for decaying noise schedules."""

    def __call__(self, age):
        raise NotImplementedError

    def _clamp(self, val, min_val=1e-10):
        if isinstance(val, torch.Tensor):
            return val.clamp(min=min_val)
        return max(val, min_val)


class ConstantSchedule(DecaySchedule):
    """σ(age) = floor.  Recovers constant-noise M2."""

    def __init__(self, floor, delta=0.0, rate=0.0):
        self.floor = floor

    def __call__(self, age):
        if isinstance(age, torch.Tensor):
            return torch.full_like(age, self.floor, dtype=torch.float).clamp(min=1e-10)
        return max(self.floor, 1e-10)


class ExponentialDecay(DecaySchedule):
    """σ(age) = floor + Δ · exp(−rate · age)"""

    def __init__(self, floor, delta, rate):
        self.floor = floor
        self.delta = delta
        self.rate = rate

    def __call__(self, age):
        if isinstance(age, torch.Tensor):
            age_f = age.float()
            return self._clamp(self.floor + self.delta * torch.exp(-self.rate * age_f))
        return self._clamp(self.floor + self.delta * np.exp(-self.rate * age))


class PowerLawDecay(DecaySchedule):
    """σ(age) = floor + Δ · age^(−rate).  Heavier tail than exponential."""

    def __init__(self, floor, delta, rate):
        self.floor = floor
        self.delta = delta
        self.rate = rate

    def __call__(self, age):
        if isinstance(age, torch.Tensor):
            age_f = age.float().clamp(min=1)  # avoid 0^(-rate)
            return self._clamp(self.floor + self.delta * age_f.pow(-self.rate))
        safe_age = max(age, 1)
        return self._clamp(self.floor + self.delta * safe_age ** (-self.rate))


class InverseLinearDecay(DecaySchedule):
    """σ(age) = floor + Δ / (1 + rate · age)"""

    def __init__(self, floor, delta, rate):
        self.floor = floor
        self.delta = delta
        self.rate = rate

    def __call__(self, age):
        if isinstance(age, torch.Tensor):
            age_f = age.float()
            return self._clamp(self.floor + self.delta / (1.0 + self.rate * age_f))
        return self._clamp(self.floor + self.delta / (1.0 + self.rate * age))


class LinearDecay(DecaySchedule):
    """σ(age) = max(floor, (floor + Δ) − rate · age).  Piecewise linear."""

    def __init__(self, floor, delta, rate):
        self.floor = floor
        self.delta = delta
        self.rate = rate
        self.sigma_max = floor + delta

    def __call__(self, age):
        if isinstance(age, torch.Tensor):
            age_f = age.float()
            raw = self.sigma_max - self.rate * age_f
            return self._clamp(torch.clamp(raw, min=self.floor))
        raw = self.sigma_max - self.rate * age
        return self._clamp(max(raw, self.floor))


# ── factory ──────────────────────────────────────────────────────────────

_SCHEDULE_REGISTRY = {
    "constant": ConstantSchedule,
    "exponential": ExponentialDecay,
    "power-law": PowerLawDecay,
    "inverse-linear": InverseLinearDecay,
    "linear": LinearDecay,
}


def make_decay_schedule(name, floor, delta, rate):
    """Construct a decay schedule by name.

    Parameters
    ----------
    name : str
        One of 'constant', 'exponential', 'power-law',
        'inverse-linear', 'linear'.
    floor : float
        Asymptotic noise level C.
    delta : float
        Transient excess noise Δ = σ_max − C.
    rate : float
        Decay rate (λ or α depending on schedule).

    Returns
    -------
    DecaySchedule
    """
    if name not in _SCHEDULE_REGISTRY:
        raise ValueError(
            f"Unknown schedule '{name}'. "
            f"Choose from: {sorted(_SCHEDULE_REGISTRY)}"
        )
    return _SCHEDULE_REGISTRY[name](floor, delta, rate)
