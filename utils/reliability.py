import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import random

def estimate_split_half_reliability(df, n_splits=100, seed=42):
    """Compute average split-half reliability over multiple random splits."""
    np.random.seed(seed)
    n_participants = df.shape[0]
    corrs = []

    for _ in range(n_splits):
        indices = np.arange(n_participants)
        np.random.shuffle(indices)
        half = n_participants // 2
        group1 = df.iloc[indices[:half], :]
        group2 = df.iloc[indices[half:], :]

        x = np.nanmean(group1.values, axis=0)
        y = np.nanmean(group2.values, axis=0)
        valid = ~(np.isnan(x) | np.isnan(y))

        if np.sum(valid) >= 2:
            r, _ = pearsonr(x[valid], y[valid])
            corrs.append(r)

    return np.mean(corrs), np.std(corrs)

def compute_itemwise_split_half_reliability(
    exps,
    criterion=1,
    n_splits=100,
    random_seed=42,
    min_isi=None,
    max_isi=None
):
    """
    Compute split-half reliability for itemwise responses.

    Args:
        exps (list of pd.DataFrame): one per participant
        criterion (int): response threshold for 'yes'
        n_splits (int): number of split-half iterations
        random_seed (int): random seed
        min_isi (int or None): if set, include only repeat trials with ISI >= this
        max_isi (int or None): if set, include only repeat trials with ISI <= this

    Returns:
        dict with reliability estimates and response matrices
    """
    random.seed(random_seed)
    np.random.seed(random_seed)

    all_signal_items = set()
    all_noise_items = set()
    participant_data_signal = []
    participant_data_noise = []

    for participant_id, df in enumerate(exps):
        yt_ids    = df['yt_id'].tolist()
        responses = df['response'].tolist()
        repeats   = df['repeat'].tolist()
        isis      = df['isi'].tolist() if 'isi' in df.columns else [None] * len(df)

        signal_row = {}
        noise_row = {}

        for yt, resp, repeat, isi in zip(yt_ids, responses, repeats, isis):
            if pd.isna(resp) or pd.isna(yt):
                continue

            is_yes = int(int(resp) > criterion)

            if repeat == 'true':
                if (min_isi is not None and isi < min_isi) or (max_isi is not None and isi > max_isi):
                    continue
                signal_row[yt] = is_yes
                all_signal_items.add(yt)

            elif repeat == 'false':
                noise_row[yt] = is_yes
                all_noise_items.add(yt)

        participant_data_signal.append(signal_row)
        participant_data_noise.append(noise_row)

    signal_df = pd.DataFrame(participant_data_signal, columns=sorted(all_signal_items)).astype("float")
    noise_df  = pd.DataFrame(participant_data_noise,  columns=sorted(all_noise_items)).astype("float")

    signal_r, signal_std = estimate_split_half_reliability(signal_df, n_splits=n_splits, seed=random_seed)
    noise_r,  noise_std  = estimate_split_half_reliability(noise_df,  n_splits=n_splits, seed=random_seed)

    return {
        'split_half_reliability': {
            'hits': (signal_r, signal_std),
            'false_alarms': (noise_r, noise_std)
        },
        'itemwise_responses': {
            'hits': signal_df,
            'false_alarms': noise_df
        }
    }

def compute_power_curve(df, n_repeats=20, n_splits=50, max_participants=None, step=5, seed=42):
    np.random.seed(seed)
    total_participants = df.shape[0]
    if max_participants is None:
        max_participants = total_participants

    sizes, means, stds = [], [], []

    for n in range(step, max_participants + 1, step):
        rs = []
        for _ in range(n_repeats):
            sample = df.sample(n=n, replace=False)
            mean_r, std_r = estimate_split_half_reliability(sample, n_splits=n_splits)
            rs.append(mean_r)

        sizes.append(n)
        means.append(np.mean(rs))
        stds.append(np.std(rs))

    return sizes, means, stds