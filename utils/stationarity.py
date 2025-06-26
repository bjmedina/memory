import numpy as np


def match_stationarity(times, scores, target, mode="closest"):
    """
    Finds the index of the closest or max stationarity score.

    Args:
        times (list): Time values.
        scores (list): Stationarity scores.
        target (float): Target score to match.
        mode (str): 'closest' (default) or 'max'.

    Returns:
        tuple: (best_index, best_time, best_score)
    """
    if mode == "max":
        idx = np.argmax(scores)
    else:
        idx = np.argmin([(target - s) ** 2 for s in scores])
    return idx, times[idx], scores[idx]
