import torch
import itertools
import numpy as np


def select_best_pair_ratio(time_avg_norms, cochleagram_norms, segment_times, segment_duration=0.5):
    """
    Selects the pair of segments that maximizes the ratio of time-averaged to full cochleagram norms.
    Filters out pairs too close in time.
    """
    time_avg_norms = time_avg_norms.cpu().numpy()
    cochleagram_norms = cochleagram_norms.cpu().numpy()
    pairs = list(itertools.combinations(range(time_avg_norms.shape[0]), 2))

    best_pair, best_ratio = None, 0
    valid_ratios, scores = [], []

    for i, j in pairs:
        if abs(segment_times[i] - segment_times[j]) >= segment_duration:
            ratio = time_avg_norms[i, j] / (cochleagram_norms[i, j] + 1e-5)
            valid_ratios.append(ratio)
            scores.append(ratio)
            if ratio > best_ratio:
                best_ratio, best_pair = ratio, (i, j)

    return best_pair, best_ratio, valid_ratios, scores


def select_best_pair_ratio_flipped(time_avg_norms, cochleagram_norms, segment_times, segment_duration=0.5):
    """
    Selects the pair of segments that maximizes the ratio of time-avg to full cochleagram norms,
    after normalizing both matrices to [0, 1].
    """
    time_avg_norms = time_avg_norms.cpu().numpy()
    cochleagram_norms = cochleagram_norms.cpu().numpy()

    cochleagram_dists_norm = (cochleagram_norms - cochleagram_norms.min()) / (cochleagram_norms.max() - cochleagram_norms.min())
    time_avg_dists_norm = (time_avg_norms - time_avg_norms.min()) / (time_avg_norms.max() - time_avg_norms.min())

    pairs = list(itertools.combinations(range(time_avg_norms.shape[0]), 2))
    best_pair, best_ratio = None, 0
    valid_ratios, scores = [], []

    for i, j in pairs:
        if abs(segment_times[i] - segment_times[j]) >= segment_duration:
            ratio = time_avg_norms[i, j] / (cochleagram_norms[i, j] + 1e-5)
            valid_ratios.append(ratio)
            scores.append(ratio)
            if ratio > best_ratio:
                best_ratio, best_pair = ratio, (i, j)

    return best_pair, best_ratio, valid_ratios, scores


def select_best_pair_ratio_flipped_torch(time_avg, cochleagrams, segment_times, segment_duration=0.5):
    """
    Torch version: Select the pair that maximizes cochleagram MSE over time-avg MSE,
    filtering out temporally adjacent pairs.
    Also returns MSE values for analysis.
    """
    device = time_avg.device
    num_clips = cochleagrams.shape[0]
    segment_times = torch.tensor(segment_times, device=device) if not torch.is_tensor(segment_times) else segment_times

    best_pair, best_ratio = None, -1e9
    valid_ratios, valid_ua_mse, valid_ta_mse = [], [], []

    for i in range(num_clips):
        for j in range(i + 1, num_clips):
            if torch.abs(segment_times[i] - segment_times[j]) >= segment_duration:
                ratio = cochleagrams[i, j] / (time_avg[i, j] + 1e-10)
                valid_ratios.append(ratio.cpu().item())
                valid_ua_mse.append(cochleagrams[i, j].cpu().item())
                valid_ta_mse.append(time_avg[i, j].cpu().item())
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_pair = (i, j)

    best_ua = cochleagrams[best_pair[0], best_pair[1]]
    best_ta = time_avg[best_pair[0], best_pair[1]]

    return best_pair, best_ratio, valid_ratios, valid_ua_mse, valid_ta_mse, best_ua, best_ta
