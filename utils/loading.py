import os
import json
import numpy as np
import pandas as pd
from scipy.stats import norm


'''
functions to help load human results
'''

def compute_dprime_from_pairs(df, selected_pairs, criterion=1, epsilon=0.0001):
    """Compute d′ using selected (nonrepeat, repeat) trial index pairs"""
    
    # Get original indices for signals and noise
    repeat_idxs = [pair[1] for pair in selected_pairs]
    nonrepeat_idxs = [pair[0] for pair in selected_pairs]

    # Signal = correct responses on repeats (hits)
    hit_responses = df.set_index("orig_index").loc[repeat_idxs, "response"].dropna()
    hits = (hit_responses.astype(int) > criterion).sum()
    n_hits = len(hit_responses)

    # Noise = incorrect responses on non-repeats (false alarms)
    fa_responses = df.set_index("orig_index").loc[nonrepeat_idxs, "response"].dropna()
    fas = (fa_responses.astype(int) > criterion).sum()
    n_fas = len(fa_responses)

    # Rates with epsilon correction
    hit_rate = min(max(hits / n_hits, epsilon), 1 - epsilon) if n_hits > 0 else np.nan
    fa_rate  = min(max(fas / n_fas, epsilon), 1 - epsilon) if n_fas > 0 else np.nan

    # Compute d'
    if np.isfinite(hit_rate) and np.isfinite(fa_rate):
        dprime = norm.ppf(hit_rate) - norm.ppf(fa_rate)
    else:
        dprime = np.nan

    return dprime, hit_rate, fa_rate

def load_results_with_exclusion(
    results_dir,
    min_dprime=1.0,
    min_trials=120,
    skip_len60=True,
    verbose=False,
    return_skipped=False,
):
    exps, seqs, fnames = load_results(
        results_dir=results_dir,
        min_trials=min_trials,
        skip_len60=skip_len60
    )

    np.random.seed(42)  # For reproducibility
    filtered_exps = []
    filtered_seqs = []
    filtered_fnames = []

    for df, seq_file, fname in zip(exps, seqs, fnames):
        df = df.copy()

        if "isi" not in df.columns or "yt_id" not in df.columns:
            if verbose:
                print(f"[{fname}] Skipped: missing required columns")
            continue

        # Keep original index to avoid KeyError
        df["orig_index"] = df.index

        # Identify repeat trials with ISI == 0
        isi0_repeat_rows = df[df["isi"] == 0.0]
        paired_indices = []

        yt_map = {}
        for i, yt in zip(df["orig_index"], df["yt_id"]):
            if yt not in yt_map:
                yt_map[yt] = []
            yt_map[yt].append(i)

        for i in isi0_repeat_rows["orig_index"]:
            yt = df.loc[i, "yt_id"]
            prior_indices = [j for j in yt_map[yt] if j < i]
            if not prior_indices:
                continue
            paired_indices.append((prior_indices[-1], i))  # (non-repeat, repeat)

        if len(paired_indices) < 2:
            if verbose:
                print(f"[{fname}] Skipped: fewer than 2 ISI=0 repeat pairs")
            continue

        # Sample and drop
        n_sample = len(paired_indices) // 2
        selected = np.random.choice(len(paired_indices), size=n_sample, replace=False)
        selected_pairs = [paired_indices[i] for i in selected]
        
        drop_ids = [j for pair in selected_pairs for j in pair]

        dprime, hit_rate, fa_rate = compute_dprime_from_pairs(df, selected_pairs)

        print(f"[{fname}] d' = {dprime:.2f} (HR={hit_rate:.2f}, FAR={fa_rate:.2f})")

        df = df[~df["orig_index"].isin(drop_ids)].drop(columns="orig_index").reset_index(drop=True)

        if dprime >= min_dprime:
    
            filtered_exps.append(df)
            filtered_seqs.append(seq_file)
            filtered_fnames.append(fname)

        if verbose:
            print(f"[{fname}] Dropped {len(drop_ids)} trials (ISI=0 pairs)")

    return filtered_exps, filtered_seqs, filtered_fnames


def load_results(results_dir, isi_pow=2, min_trials=120, skip_len60=True):
    """Load and filter experiment result CSVs."""
    files = sorted(
        [f for f in os.listdir(results_dir) if f.endswith(".csv")],
        key=lambda fn: os.path.getctime(os.path.join(results_dir, fn))
    )
    exps, seqs, fnames = [], [], []
    for fn in files:
        df = pd.read_csv(os.path.join(results_dir, fn))
        main = df[df.stim_type == "main"]
        seq_file = main.sequence_file.iloc[0].split("/")[-1]
        if len(main) < min_trials: continue
        if "tol0" in seq_file: continue
        exps.append(main); seqs.append(seq_file); fnames.append(fn)
    return exps, seqs, fnames

def load_results_with_isi0_dprime_exclusion(
    results_dir,
    min_dprime=1.0,
    min_trials=120,
    skip_len60=True,
    criterion=1,
    exclusion_isi=0,
    exclusion_frac=0.5,
    seed=42,
):
    np.random.seed(seed)

    files = sorted(
        [f for f in os.listdir(results_dir) if f.endswith(".csv")],
        key=lambda fn: os.path.getctime(os.path.join(results_dir, fn)),
    )

    exps, seqs, fnames = [], [], []

    for fn in files:
        df = pd.read_csv(os.path.join(results_dir, fn))
        main = df[df.stim_type == "main"].copy()

        if len(main) < min_trials:
            continue

        seq_file = main.sequence_file.iloc[0].split("/")[-1]
        if "tol0" in seq_file and skip_len60:
            continue

        yt_ids = main["yt_id"].tolist()
        responses = main["response"].tolist()
        repeats = main["repeat"].tolist()

        # === Track prior occurrences for ISI estimation ===
        yt_to_last_idx = {}
        isi0_trials = []

        for i, (yt, resp, repeat) in enumerate(zip(yt_ids, responses, repeats)):
            if pd.isna(resp) or pd.isna(yt) or repeat != "true":
                continue

            prev_idx = yt_to_last_idx.get(yt)
            if prev_idx is not None:
                isi = i - prev_idx - 1
                if isi == exclusion_isi:
                    isi0_trials.append((main.index[i], int(int(resp) > criterion)))

            yt_to_last_idx[yt] = i

        if len(isi0_trials) < 2:
            continue

        n_sample = int(np.floor(exclusion_frac * len(isi0_trials)))
        selected = np.random.choice(len(isi0_trials), size=n_sample, replace=False)
        selected_trials = [isi0_trials[i] for i in selected]

        # === Compute d′ from this subsample ===
        hits = sum(resp for _, resp in selected_trials)
        n_signal = len(selected_trials)

        fa_trials = main[(main["repeat"] == "false") & (~main["response"].isna())]
        fas = fa_trials["response"].apply(lambda r: int(int(r) > criterion))
        n_noise = len(fas)
        fa_total = fas.sum()

        hit_rate = hits / n_signal if n_signal > 0 else np.nan
        fa_rate = fa_total / n_noise if n_noise > 0 else np.nan
        dprime = compute_dprime(hit_rate, fa_rate) if np.isfinite(hit_rate) and np.isfinite(fa_rate) else np.nan

        if not np.isfinite(dprime) or dprime < min_dprime:
            continue  # Exclude participant

        # === Remove sampled trials from the main df ===
        rows_to_drop = [idx for idx, _ in selected_trials]
        main = main.drop(index=rows_to_drop).reset_index(drop=True)

        exps.append(main)
        seqs.append(seq_file)
        fnames.append(fn)

    return exps, seqs, fnames


def load_results_with_isi0_exclusion(
    results_dir,
    isi_pow=2,
    min_trials=120,
    skip_len60=True,
    exclusion_isi=0,
    exclusion_frac=0.5,
    exclusion_min_acc=0.9,
    criterion=1,
    seed=42,
):
    np.random.seed(seed)

    files = sorted(
        [f for f in os.listdir(results_dir) if f.endswith(".csv")],
        key=lambda fn: os.path.getctime(os.path.join(results_dir, fn)),
    )

    exps, seqs, fnames = [], [], []

    for fn in files:
        df = pd.read_csv(os.path.join(results_dir, fn))
        main = df[df.stim_type == "main"].copy()
        if len(main) < min_trials:
            continue
        seq_file = main.sequence_file.iloc[0].split("/")[-1]
        if "tol0" in seq_file and skip_len60:
            continue

        # === ISI=0 exclusion logic ===
        yt_ids = main["yt_id"].tolist()
        responses = main["response"].tolist()
        repeats = main["repeat"].tolist()

        isi0_trials = []

        for i, (yt, resp, repeat) in enumerate(zip(yt_ids, responses, repeats)):
            if repeat != "true" or pd.isna(resp) or pd.isna(yt):
                continue
            try:
                j = yt_ids[:i].index(yt)
                isi = i - j - 1
                if isi == exclusion_isi:
                    isi0_trials.append((main.index[i], int(resp) > criterion))
            except ValueError:
                continue

        n_isi0 = len(isi0_trials)
        if n_isi0 < 2:
            continue

        n_sample = int(np.floor(exclusion_frac * n_isi0))
        selected_indices = np.random.choice(n_isi0, size=n_sample, replace=False)
        selected = [isi0_trials[i] for i in selected_indices]

        acc = np.mean([resp for _, resp in selected])
        if acc < exclusion_min_acc:
            continue  # exclude participant

        # drop these sampled ISI=0 trials from analysis
        exclude_rows = [isi0_trials[i][0] for i in selected_indices]
        main = main.drop(index=exclude_rows).reset_index(drop=True)

        exps.append(main)
        seqs.append(seq_file)
        fnames.append(fn)

    return exps, seqs, fnames

def remove_sequences_with_len60(seq_dir):
    """Remove entries containing 'len60' from unused.json and used.json."""
    for sub in ("unused", "used"):
        path = os.path.join(seq_dir, sub, f"{sub}.json")
        data = json.load(open(path))
        filtered = [f for f in data if "len60" not in f]
        json.dump(sorted(filtered), open(path, "w"), indent=2)

def move_sequences_to_used(seq_dir, seqs_used):
    """Move used sequence filenames from unused.json to used.json."""
    u_path = os.path.join(seq_dir, "unused", "unused.json")
    z_path = os.path.join(seq_dir, "used",   "used.json")
    unused = json.load(open(u_path)); used = json.load(open(z_path))
    seqs = [os.path.basename(s) for s in seqs_used]
    new_unused = [s for s in unused if s not in seqs]
    new_used = sorted(set(used + seqs))
    json.dump(sorted(new_unused), open(u_path, "w"), indent=2)
    json.dump(new_used,         open(z_path, "w"), indent=2)