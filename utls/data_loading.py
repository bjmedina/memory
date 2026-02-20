"""
Experiment data loading, filtering, and batch management for human analysis.

Consolidates the repeated data-loading boilerplate from the analysis notebooks
into reusable functions with configurable task, delay, and ISI parameters.
"""

import os
import re
import json
import random
import numpy as np
import pandas as pd
from collections import defaultdict


# ── task registry ────────────────────────────────────────────────────

TASKS_MULTI = ["env-sounds", "glob-music", "atexts", "nhs-region-len120"]
TASKS_SINGLE = [
    "ind-nature-len120",
    "global-music-len120",
    "atexts-len120",
    "nhs-region-len120",
]

SEQS_PATHS = {
    # multi-ISI task names
    "env-sounds": "mem_exp_ind-nature_2025",
    "glob-music": "global-music-2025-n_80",
    "atexts": "mem_exp_atexts_2025",
    "nhs-region-len120": "nhs-region-n_80",
    # single-ISI task names
    "ind-nature-len120": "mem_exp_ind-nature_2025",
    "global-music-len120": "global-music-2025-n_80",
    "atexts-len120": "mem_exp_atexts_2025",
}

HR_TASK_NAMES = {
    "env-sounds": "Industrial and Nature",
    "glob-music": "Globalized Music",
    "atexts": "Auditory Textures",
    "nhs-region-len120": "Natural History of Song",
    "ind-nature-len120": "Industrial and Nature",
    "global-music-len120": "Globalized Music",
    "atexts-len120": "Auditory Textures",
}


# ── path builders ────────────────────────────────────────────────────

def _stim_base(task):
    """Base stimulus directory for a task."""
    return f"/mindhive/mcdermott/www/mturk_stimuli/bjmedina/{SEQS_PATHS[task]}"


def results_path_multi(task, delay=None):
    """Build the results directory path for a multi-ISI experiment."""
    suffix = f"len120_multi_delay_{delay}s" if delay else "len120_multi"
    return f"/mindhive/mcdermott/www/bjmedina/experiments/{task}/results/{task}/{suffix}"


def results_path_single(task, isi):
    """Build the results directory path for a single-ISI experiment."""
    return f"/mindhive/mcdermott/www/bjmedina/experiments/bolivia_2025/results/isi_{isi}/{task}"


def sequences_path_multi(task, delay=None):
    """Build the sequences directory path for a multi-ISI experiment."""
    suffix = f"len120_multi_delay_{delay}s" if delay else "len120_multi"
    return f"{_stim_base(task)}/sequences/{suffix}/"


def sequences_path_single(task, isi):
    """Build the sequences directory path for a single-ISI experiment."""
    return f"{_stim_base(task)}/sequences/isi_{isi}/len120/"


# ── core loading ─────────────────────────────────────────────────────

def load_and_filter(
    results_dir,
    load_fn,
    min_dprime=2,
    min_trials=120,
    skip_len60=True,
    return_skipped=False,
):
    """
    Load experiment data and apply standard filters.

    Parameters
    ----------
    results_dir : str
        Directory containing CSV result files.
    load_fn : callable
        The actual loading function (e.g., load_results_with_exclusion_no_dropping
        from utls.loading).
    min_dprime : float
        Minimum d' for participant inclusion.
    min_trials : int
        Minimum trial count.
    skip_len60 : bool
        Skip len60 sequences.
    return_skipped : bool
        Whether to return skipped data.

    Returns
    -------
    If return_skipped: (exps, seqs, fnames, skipped_exps, skipped_seqs, skipped_fnames)
    Else: (exps, seqs, fnames)
    """
    return load_fn(
        results_dir,
        min_dprime=min_dprime,
        min_trials=min_trials,
        skip_len60=skip_len60,
        verbose=False,
        return_skipped=return_skipped,
    )


# ── deduplication ────────────────────────────────────────────────────

def deduplicate_by_sequence(exps, seqs, fnames):
    """
    Keep one randomly-chosen participant per unique sequence.

    Returns
    -------
    exps, seqs, fnames : filtered lists.
    """
    seq_to_indices = defaultdict(list)
    for i, s in enumerate(seqs):
        seq_to_indices[s].append(i)

    keep = [random.choice(idxs) for idxs in seq_to_indices.values()]
    return (
        [exps[i] for i in keep],
        [seqs[i] for i in keep],
        [fnames[i] for i in keep],
    )


# ── batch filtering ──────────────────────────────────────────────────

def filter_complete_batches(exps, seqs, fnames, batch_size=8):
    """
    Keep only participants whose sequences belong to complete batches.

    A "complete batch" has exactly `batch_size` consecutive sequences
    (e.g., seq1-8 is batch 0, seq9-16 is batch 1, etc.).

    Returns
    -------
    exps, seqs, fnames, info : filtered lists + info dict.
    """
    seqnums = [int(re.search(r"seq(\d+)", s).group(1)) for s in seqs]

    batch_to_seqnums = defaultdict(set)
    for n in seqnums:
        batch_id = (n - 1) // batch_size
        batch_to_seqnums[batch_id].add(n)

    complete_batches = {
        b
        for b, nums in batch_to_seqnums.items()
        if nums == set(range(b * batch_size + 1, b * batch_size + batch_size + 1))
    }

    keep = [
        i for i, n in enumerate(seqnums) if (n - 1) // batch_size in complete_batches
    ]

    info = {
        "complete_batches": sorted(complete_batches),
        "n_kept": len(keep),
        "n_total": len(seqs),
    }

    return (
        [exps[i] for i in keep],
        [seqs[i] for i in keep],
        [fnames[i] for i in keep],
        info,
    )


# ── high-level loaders ───────────────────────────────────────────────

def load_multi_isi(
    task,
    load_fn,
    delay=None,
    min_dprime=2,
    min_trials=120,
    batch_size=8,
    deduplicate=True,
    filter_batches=True,
):
    """
    Load a multi-ISI experiment with standard preprocessing.

    Parameters
    ----------
    task : str
        One of TASKS_MULTI (e.g., 'atexts', 'env-sounds').
    load_fn : callable
        The raw loading function (load_results_with_exclusion_no_dropping).
    delay : int or None
        Delay condition in seconds (None = no delay).
    min_dprime : float
        Minimum d' for inclusion.
    min_trials : int
        Minimum trial count.
    batch_size : int
        Batch size for filtering.
    deduplicate : bool
        Remove duplicate sequences.
    filter_batches : bool
        Keep only complete batches.

    Returns
    -------
    dict with: exps, seqs, fnames, task, hr_name, delay, batch_info.
    """
    rdir = results_path_multi(task, delay=delay)
    exps, seqs, fnames, _, _, _ = load_fn(
        rdir,
        min_dprime=min_dprime,
        min_trials=min_trials,
        skip_len60=True,
        verbose=False,
        return_skipped=True,
    )

    if deduplicate:
        exps, seqs, fnames = deduplicate_by_sequence(exps, seqs, fnames)

    batch_info = None
    if filter_batches:
        exps, seqs, fnames, batch_info = filter_complete_batches(
            exps, seqs, fnames, batch_size=batch_size
        )

    return {
        "exps": exps,
        "seqs": seqs,
        "fnames": fnames,
        "task": task,
        "hr_name": HR_TASK_NAMES.get(task, task),
        "delay": delay,
        "batch_info": batch_info,
        "N": len(exps),
    }


def load_single_isi(
    task,
    isi,
    load_fn,
    min_dprime=2,
    min_trials=120,
):
    """
    Load a single-ISI experiment with standard preprocessing.

    Parameters
    ----------
    task : str
        One of TASKS_SINGLE (e.g., 'atexts-len120').
    isi : int
        The ISI value for this experiment.
    load_fn : callable
        The raw loading function.

    Returns
    -------
    dict with: exps, seqs, fnames, task, hr_name, isi, N.
    """
    rdir = results_path_single(task, isi)
    exps, seqs, fnames = load_fn(
        rdir,
        min_dprime=min_dprime,
        min_trials=min_trials,
        skip_len60=True,
        verbose=False,
        return_skipped=False,
    )

    return {
        "exps": exps,
        "seqs": seqs,
        "fnames": fnames,
        "task": task,
        "hr_name": HR_TASK_NAMES.get(task, task),
        "isi": isi,
        "N": len(exps),
    }


# ── figure save directory ────────────────────────────────────────────

def make_save_dir(base, task, sub=None):
    """
    Create and return a figure save directory.

    Parameters
    ----------
    base : str
        Base figures directory.
    task : str
        Task name (used to build subdirectory).
    sub : str or None
        Additional subdirectory (e.g., 'multi-isi_2s').

    Returns
    -------
    str : the directory path (created if it doesn't exist).
    """
    safe_name = task.lower().replace(" ", "_")
    parts = [base]
    if sub:
        parts.append(sub)
    parts.append(safe_name)
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path
