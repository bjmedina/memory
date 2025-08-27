import numpy as np
import pandas as pd
from scipy.stats import norm
from collections import defaultdict

def compute_dprime(hit_rate, fa_rate):
    hit_rate = np.clip(hit_rate, 1e-5, 1 - 1e-5)
    fa_rate = np.clip(fa_rate, 1e-5, 1 - 1e-5)
    return norm.ppf(hit_rate) - norm.ppf(fa_rate)


def recompute_dprime_by_isi(exps, criterion=1):
    hit_counts = defaultdict(int)
    fa_counts  = defaultdict(int)
    signal_counts = defaultdict(int)
    noise_counts  = defaultdict(int)

    for df in exps:
        seen_yt_ids = {}
        yt_ids = df['yt_id'].tolist()
        responses = df['response'].tolist()
        repeats = df['repeat'].tolist()

 
        for i, (yt, resp, repeat) in enumerate(zip(yt_ids, responses, repeats )):
            if pd.isna(resp) or pd.isna(yt):
                continue

            is_yes = int(int(resp) > criterion)

            if repeat == 'true':
                j = yt_ids[:i].index(yt)
                isi = i - j - 1

                if isi not in [-1,0,1,2,3, 4, 8, 16, 32, 64]:
                    continue

                #print(j, i, isi)
                #print(yt_ids[j], yt)
                signal_counts[isi] += 1
                hit_counts[isi] += is_yes
            elif repeat == 'false':
                noise_counts[-1] += 1  # ISI=-1 for noise trials
                fa_counts[-1] += is_yes

                #seen_yt_ids[yt] = i  # store first appearance

        #print("--")

    # Build results
    results = []
    all_isi_vals = sorted(set(signal_counts) | set(noise_counts))
    for isi in all_isi_vals:
        hits = hit_counts[isi]
        fas  = fa_counts[-1]
        n_signal = signal_counts[isi]
        n_noise  = noise_counts[-1]

        hit_rate = hits / n_signal if n_signal > 0 else np.nan
        fa_rate  = fas  / n_noise  if n_noise  > 0 else np.nan
        dprime_val = compute_dprime(hit_rate, fa_rate) #if np.isfinite(hit_rate) and np.isfinite(fa_rate) else np.nan

        results.append({
            'isi': isi,
            'hits': hits,
            'false_alarms': fas,
            'n_signal': n_signal,
            'n_noise': n_noise,
            'hit_rate': hit_rate,
            'fa_rate': fa_rate,
            'd_prime': dprime_val
        })

    return pd.DataFrame(results).sort_values(by='isi')


def recompute_dprime_by_isi_per_subject(exps, criterion=1):
    allowed_isi = {-1, 0, 1, 2, 3, 4, 8, 16, 32, 64}
    all_results = []

    for subj_idx, df in enumerate(exps):
        hit_counts = defaultdict(int)
        fa_counts  = defaultdict(int)
        signal_counts = defaultdict(int)
        noise_counts  = defaultdict(int)

        yt_ids = df['yt_id'].tolist()
        responses = df['response'].tolist()
        repeats = df['repeat'].tolist()

        for i, (yt, resp, repeat) in enumerate(zip(yt_ids, responses, repeats)):
            if pd.isna(resp) or pd.isna(yt):
                continue

            is_yes = int(int(resp) > criterion)

            if repeat == 'true':
                try:
                    j = yt_ids[:i].index(yt)
                    isi = i - j - 1
                except ValueError:
                    continue  # yt not found in earlier trials

                if isi not in allowed_isi:
                    continue

                signal_counts[isi] += 1
                hit_counts[isi] += is_yes

            elif repeat == 'false':
                isi = -1
                noise_counts[isi] += 1
                fa_counts[isi] += is_yes

        # Aggregate per-ISI results for this subject
        for isi in sorted(signal_counts.keys() | noise_counts.keys()):
            n_signal = signal_counts[isi]
            n_noise  = noise_counts.get(-1, 0)  # all noise trials pooled under -1
            hits = hit_counts[isi]
            fas  = fa_counts.get(-1, 0)

            hit_rate = hits / n_signal if n_signal > 0 else np.nan
            fa_rate  = fas  / n_noise  if n_noise  > 0 else np.nan
            d_prime = compute_dprime(hit_rate, fa_rate) if np.isfinite(hit_rate) and np.isfinite(fa_rate) else np.nan

            all_results.append({
                'subject': subj_idx,
                'isi': isi,
                'hits': hits,
                'false_alarms': fas,
                'n_signal': n_signal,
                'n_noise': n_noise,
                'hit_rate': hit_rate,
                'fa_rate': fa_rate,
                'd_prime': d_prime
            })

    return pd.DataFrame(all_results).sort_values(by=['subject', 'isi'])