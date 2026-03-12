#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utls.roc_utils — ROC, AUC, and d′ utilities
"""

import numpy as np
from sklearn.metrics import roc_curve, auc
from scipy.stats import norm
import matplotlib.pyplot as plt


def roc_from_arrays(hit_scores, fa_scores, score_type="distance"):
    """
    Compute ROC curve and AUC given hit and FA scores.
    """
    if hit_scores.size == 0 or fa_scores.size == 0:
        return None

    y_true = np.concatenate([
        np.ones_like(hit_scores),
        np.zeros_like(fa_scores)
    ])
    y_score = np.concatenate([hit_scores, fa_scores])

    if score_type == "distance":
        y_score = -y_score
    elif score_type != "likelihood":
        raise ValueError(f"Unknown score_type: {score_type}")

    fpr, tpr, _ = roc_curve(y_true, y_score)
    return fpr, tpr, auc(fpr, tpr)

def roc_from_arrays_with_threshold(hit_scores, fa_scores, score_type="distance"):
    """
    Compute ROC curve and AUC given hit and FA scores.
    """
    if hit_scores.size == 0 or fa_scores.size == 0:
        return None

    y_true = np.concatenate([
        np.ones_like(hit_scores),
        np.zeros_like(fa_scores)
    ])
    y_score = np.concatenate([hit_scores, fa_scores])

    if score_type == "distance":
        y_score = -y_score
    elif score_type != "likelihood":
        raise ValueError(f"Unknown score_type: {score_type}")

    fpr, tpr, threshold = roc_curve(y_true, y_score)
    return fpr, tpr, threshold

def roc_for_isi(run_data, isi_value):
    hits_raw = run_data["isi_hit_dists"].get(isi_value, [])
    if not hits_raw:
        return None

    hits = np.array([d for (d, t) in hits_raw], float)

    # Use all FAs (pooled across time) as comparison set
    fas = np.asarray(run_data["fas"], float)

    score_type = run_data.get("score_type", "distance")
    return roc_from_arrays(hits, fas, score_type=score_type)


def plot_roc(ax, fpr, tpr, label):
    """Plot ROC curve on given axis."""
    ax.plot(fpr, tpr, label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def auroc_to_dprime(auroc):
    """Convert AUROC to d′ via z-transform rule."""
    auroc = np.clip(auroc, 1e-8, 1 - 1e-8)
    return np.sqrt(2) * norm.ppf(auroc)