"""
Toy experiment generators for fitting individual noise parameters.

The memory model uses a three-regime noise schedule:
  - sigma0: applied at ISI = 0 (immediate repeats)
  - sigma1: applied at 0 < ISI < t_step (short delays, typically ISI 1-4)
  - sigma2: applied at ISI >= t_step (long delays, typically ISI 8-64)

By creating experiments where all repeats occur at a specific ISI, we can
isolate and fit each sigma parameter independently:
  1. sigma0 is fit using ISI-0-only experiments
  2. sigma1 is fit using ISI 1-4 experiments  (with sigma0 fixed)
  3. sigma2 is fit using ISI 8-64 experiments (with sigma0 and sigma1 fixed)
"""

import random


def make_isi_n_block_experiment(stimuli, isi):
    """
    Build an experiment sequence where every stimulus repeats at exactly
    the given ISI (number of intervening items).

    Strategy: group stimuli into blocks of size (isi + 1).  Within each
    block the first (isi + 1) items are first presentations and the next
    (isi + 1) are repeats **in the same order**, which guarantees each
    repeat is separated from its first presentation by exactly *isi*
    intervening items.

    Examples
    --------
    ISI=0, stimuli=[A,B,C]  -> [A, A, B, B, C, C]
    ISI=1, stimuli=[A,B]    -> [A, B, A, B]
    ISI=2, stimuli=[A,B,C]  -> [A, B, C, A, B, C]
    ISI=4, stimuli=[A,B,C,D,E] -> [A,B,C,D,E, A,B,C,D,E]

    Parameters
    ----------
    stimuli : list
        Stimulus identifiers (file paths, etc.).
    isi : int
        Target number of intervening items between first and second
        presentation of each stimulus.

    Returns
    -------
    list
        Experiment sequence.
    """
    if isi == 0:
        seq = []
        for s in stimuli:
            seq.extend([s, s])
        return seq

    block_size = isi + 1
    seq = []

    for i in range(0, len(stimuli), block_size):
        block = stimuli[i : i + block_size]
        if len(block) < block_size:
            # incomplete block — skip to avoid wrong ISI
            break
        seq.extend(block)   # first presentations
        seq.extend(block)   # repeats (same order → ISI = block_size - 1)

    return seq


def make_toy_experiment_list(
    stimulus_pool, isi, n_experiments=20, k_stimuli=10, seed=None
):
    """
    Create multiple toy experiments for a given ISI value.

    Each experiment draws *k_stimuli* from the pool (without replacement),
    shuffles them, and arranges them so every repeat occurs at exactly
    *isi* intervening items.

    Parameters
    ----------
    stimulus_pool : list
        All available stimulus paths.
    isi : int
        Target ISI for every repeat in the experiment.
    n_experiments : int
        Number of independent experiments to generate.
    k_stimuli : int
        Distinct stimuli sampled per experiment.  Must be >= isi + 1 so
        that at least one complete block can be formed.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    list[list]
        Each inner list is one experiment sequence.
    """
    rng = random.Random(seed)
    pool = list(stimulus_pool)

    # ensure we can form at least one block
    min_k = isi + 1
    if k_stimuli < min_k:
        k_stimuli = min_k

    # ensure we don't sample more than available
    k_stimuli = min(k_stimuli, len(pool))

    exp_list = []
    for _ in range(n_experiments):
        rng.shuffle(pool)
        stimuli = pool[:k_stimuli]
        exp = make_isi_n_block_experiment(stimuli, isi)
        if len(exp) > 0:
            exp_list.append(exp)

    return exp_list


def make_multi_isi_toy_experiments(
    stimulus_pool,
    isi_values,
    n_experiments_per_isi=20,
    k_stimuli=10,
    seed=None,
):
    """
    Create toy experiment lists for multiple ISI values.

    Parameters
    ----------
    stimulus_pool : list
        All available stimulus paths.
    isi_values : list[int]
        ISI values to generate experiments for (e.g. [1, 2, 4]).
    n_experiments_per_isi : int
        Experiments generated per ISI value.
    k_stimuli : int
        Stimuli per experiment (adjusted upward if smaller than isi + 1).
    seed : int or None
        Base random seed; each ISI gets a distinct derived seed.

    Returns
    -------
    dict[int, list[list]]
        Mapping from ISI value to its list of experiment sequences.
    """
    base_seed = seed if seed is not None else 42
    experiments_by_isi = {}

    for i, isi in enumerate(isi_values):
        experiments_by_isi[isi] = make_toy_experiment_list(
            stimulus_pool,
            isi=isi,
            n_experiments=n_experiments_per_isi,
            k_stimuli=k_stimuli,
            seed=base_seed + i * 1000,
        )

    return experiments_by_isi
