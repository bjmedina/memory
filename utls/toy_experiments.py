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

For sigma2 in particular, per-ISI toy experiments can be very long (ISI=64
requires ~130 trials per experiment).  ``make_compact_multi_isi_sequences``
generates shorter multi-ISI sequences using ``ISISequence`` +
``StimulusManager``, interleaving all target ISIs in a single compact sequence
(~78 trials with ~5 pairs per ISI).
"""

import random

from utils.sequence_utils import ISISequence, StimulusManager


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


# ── multi-ISI helpers ─────────────────────────────────────────────────


def infer_trial_isis(sequence):
    """Return the ISI for each repeat (target) trial in a sequence.

    Assumes each stimulus appears at most twice.  ISI = repeat_position -
    first_position - 1.

    Parameters
    ----------
    sequence : list
        Ordered stimulus identifiers (file paths, etc.).

    Returns
    -------
    list[int]
        One ISI value per repeat trial, in sequence order.
        First-presentation (foil) trials are not included.
    """
    first_seen = {}
    trial_isis = []
    for i, stim in enumerate(sequence):
        if stim in first_seen:
            trial_isis.append(i - first_seen[stim] - 1)
        else:
            first_seen[stim] = i
    return trial_isis

def make_high_diversity_sequences(
    stimulus_pool, isi_values, n_sequences, length, min_pairs_per_isi=5, seed=42,
):
    """Like make_compact_multi_isi_sequences but each sequence draws from a
    DIFFERENT slice of the full stimulus pool (no overlap between seqs)."""
    isi_values = list(isi_values)
    isi_conditions = [-1] + isi_values
    unique_per_seq = length // 3 * 2  # stimuli needed per sequence
    total_needed = unique_per_seq * n_sequences

    if len(stimulus_pool) < total_needed:
        print(f"WARNING: pool ({len(stimulus_pool)}) < needed ({total_needed}), "
              f"some overlap will occur")

    # Stage 1: generate ISI patterns (same as original)
    isi_seq = ISISequence(length=length, isi_values=isi_conditions, seed=seed)
    isi_seq.generate_n(n=n_sequences, min_pairs_per_isi=min_pairs_per_isi)

    # Stage 2: assign DIFFERENT stimuli to each sequence
    rng = random.Random(seed + 99)
    pool = list(stimulus_pool)
    rng.shuffle(pool)

    experiment_list = []
    isi_keys = []

    for j in range(n_sequences):
        seq, pairs = isi_seq.get_sequence_and_isi_pairings(j)

        # Each sequence gets its own slice of the pool
        start = (j * unique_per_seq) % len(pool)
        end = start + unique_per_seq
        if end <= len(pool):
            seq_pool = pool[start:end]
        else:
            # Wrap around
            seq_pool = pool[start:] + pool[:end - len(pool)]

        # Use StimulusManager for a single sequence with this unique pool
        sm = StimulusManager(
            stimulus_ids=seq_pool,
            isi_values=isi_conditions,
            length=length,
            seed=seed + j * 1000,
            shuffle=True,
        )
        sm.get_assignments_from_pairs(pairs, seq=seq)
        experiment_list.append(sm.assignments[0])
        isi_keys.append(sm.seqs[0])

    return experiment_list, isi_keys

def make_compact_multi_isi_sequences(
    stimulus_pool,
    isi_values=(8, 16, 32, 64),
    n_sequences=10,
    length=78,
    min_pairs_per_isi=5,
    seed=42,
):
    """
    Generate compact multi-ISI experiment sequences.

    Uses :class:`ISISequence` to create abstract ISI patterns and
    :class:`StimulusManager` to assign actual stimuli from *stimulus_pool*.
    Each returned sequence contains interleaved repeat pairs at every
    requested ISI, keeping total length much shorter than equivalent
    per-ISI toy experiments (especially for large ISIs like 64).

    Parameters
    ----------
    stimulus_pool : list
        Available stimulus identifiers (file paths).  Must contain at
        least ``length // 3 * 2`` entries.
    isi_values : sequence of int
        Positive ISI conditions to include (e.g. ``[8, 16, 32, 64]``).
        The non-repeat condition ``-1`` is added automatically.
    n_sequences : int
        Number of independent sequences to generate.
    length : int
        Trials per sequence.  Must be divisible by 3 and large enough
        to accommodate the largest ISI (>= max(isi_values) + 2).
    min_pairs_per_isi : int
        Minimum repeat pairs per ISI condition in each sequence.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    experiment_list : list[list]
        Each inner list is a sequence of stimulus identifiers, directly
        compatible with ``run_experiment_scores``.
    isi_keys : list[list[int]]
        ISI label for every position in each sequence (``-1`` for
        non-repeats, positive int for the second presentation of a
        repeat pair).  Useful for splitting model hits by ISI.
    """
    isi_values = list(isi_values)

    if length % 3 != 0:
        raise ValueError(f"length must be divisible by 3, got {length}")

    max_isi = max(isi_values)
    if length < max_isi + 2:
        raise ValueError(
            f"length {length} is too short for ISI={max_isi} "
            f"(need at least {max_isi + 2})"
        )

    unique_stim_needed = length // 3 * 2
    if len(stimulus_pool) < unique_stim_needed:
        raise ValueError(
            f"stimulus_pool has {len(stimulus_pool)} items but "
            f"{unique_stim_needed} are needed for length={length}"
        )

    # ISI conditions including the non-repeat marker
    isi_conditions = [-1] + isi_values

    # Stage 1: generate abstract ISI patterns
    isi_seq = ISISequence(length=length, isi_values=isi_conditions, seed=seed)
    isi_seq.generate_n(
        n=n_sequences,
        min_pairs_per_isi=min_pairs_per_isi,
    )

    # Stage 2: assign stimuli to positions
    sm = StimulusManager(
        stimulus_ids=stimulus_pool[:unique_stim_needed],
        isi_values=isi_conditions,
        length=length,
        seed=seed + 1,
        shuffle=True,
    )

    for j in range(n_sequences):
        seq, pairs = isi_seq.get_sequence_and_isi_pairings(j)
        sm.get_assignments_from_pairs(pairs, seq=seq)

    experiment_list = list(sm.assignments)
    isi_keys = list(sm.seqs)

    return experiment_list, isi_keys
