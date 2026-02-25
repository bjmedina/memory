# Refactor guide for sequence-generation notebooks

This guide summarizes low-risk ways to make these notebooks easier to read and maintain while preserving current behavior.

## Notebooks reviewed

- `BOLIVIA_SEQUENCE_GENERATION-SINGLE-ISI.ipynb`
- `stim_manager_test_exemplars-MULTI_ISI.ipynb`

## Main opportunities

1. **Move class definitions out of notebooks**
   - Both notebooks embed large classes (`ISISequence`, `StimulusManager`) and plotting/export helpers.
   - Extract into a Python module, e.g. `src/memory/stim_sequence.py`, then import in notebooks.

2. **Separate "core logic" from "analysis/plots"**
   - Keep generation and validation methods together.
   - Move plotting helpers into a second class or separate functions in `plots.py`.

3. **Use typed config objects for experiment parameters**
   - Replace scattered globals (`len_exp`, `ISI_conditions`, seeds, paths) with dataclasses.
   - Makes each run reproducible and easier to compare.

4. **Create one shared save/export utility**
   - `save_all_sequences(...)` appears in notebook code.
   - Move it to a small reusable utility function with a stable signature.

5. **Make notebook cells thinner**
   - Keep notebooks to: imports, config, one generation call, one validation call, and optional plotting.
   - Avoid re-defining classes and commented-out historical code in active notebooks.

## Suggested module layout

```text
src/memory/
  sequence_config.py       # dataclasses for run settings
  sequence_generator.py    # ISISequence + StimulusManager core logic
  sequence_metrics.py      # non-plot diagnostics
  sequence_plots.py        # plotting utilities
  sequence_io.py           # save/load JSON + metadata
```

## Minimal API to target

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class SequenceConfig:
    length: int
    isi_values: list[int]
    seed: int
    n_sequences: int
    min_pairs_per_isi: int = 2

class StimulusManager:
    def __init__(self, stimulus_ids: list[str], config: SequenceConfig, shuffle: bool = True):
        ...

    def generate(self):
        """Generate a single sequence assignment."""
        ...

    def generate_n(self):
        """Generate config.n_sequences assignments."""
        ...

    def validate(self) -> dict:
        """Return consistency checks and summary metrics."""
        ...
```

## Refactor sequence (safe, incremental)

1. Copy existing class code into a `.py` file unchanged.
2. Add unit-style smoke tests for generation + uniqueness checks.
3. Update notebooks to import the new module, keeping output plots identical.
4. Split plotting methods from core class after parity is confirmed.
5. Remove duplicated/legacy class definitions from notebooks.

## Readability conventions that help in notebooks

- Keep one concept per cell (config, load stimuli, generate, validate, plot).
- Prefer short named helpers over long inline loops.
- Use one consistent naming style (`isi_values`, `n_sequences`, `seq_length`).
- Keep all filesystem paths in one config cell.
- Replace large commented blocks with a short markdown note + link to git history.

## Optional quality guardrails

- Add a `validate()` method that checks:
  - no duplicate sequence IDs
  - expected repeat/non-repeat ratios
  - ISI histogram balance and per-position spread
- Add deterministic regression checks by fixing `seed` and snapshotting summary metrics.
- Add one `make`/script entrypoint for generation to avoid rerunning many notebook cells.

