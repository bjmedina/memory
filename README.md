# Prior-Guided Drift-Diffusion Model of Auditory Recognition Memory

A computational framework for modeling how auditory memory traces degrade over time. Stored representations evolve via two competing forces: **stochastic diffusion** (random noise) and **deterministic drift** toward a learned prior over natural sounds. The model is fit to human behavioral data from serial sound recognition experiments and reproduces both aggregate forgetting curves and item-level variation in memorability.

**Authors:** Bryan J. Medina, Lakshmi N. Govindarajan, Ila R. Fiete & Josh H. McDermott

---

## Table of Contents

- [Scientific Background](#scientific-background)
- [Model Overview](#model-overview)
  - [Core Formulation](#core-formulation)
  - [The Two Main Models](#the-two-main-models)
  - [Recognition Decision](#recognition-decision)
  - [Additional Model Variants](#additional-model-variants)
- [Repository Structure](#repository-structure)
  - [File Activity Audit](#file-activity-audit)
- [Installation](#installation)
  - [Dependencies](#dependencies)
  - [External Libraries](#external-libraries)
- [Quickstart](#quickstart)
  - [2D Sandbox (No External Dependencies)](#2d-sandbox-no-external-dependencies)
  - [Full Model with Auditory Textures](#full-model-with-auditory-textures)
- [Usage](#usage)
  - [Running a Single Simulation](#running-a-single-simulation)
  - [Grid Search over Parameters](#grid-search-over-parameters)
  - [SLURM Cluster Execution](#slurm-cluster-execution)
  - [Merging Grid Search Results](#merging-grid-search-results)
  - [Parameter Fitting](#parameter-fitting)
- [Analysis Pipeline](#analysis-pipeline)
  - [Computing d' from Simulations](#computing-d-from-simulations)
  - [Human–Model Comparison](#humanmodel-comparison)
  - [Grid Search Functional Characterization](#grid-search-functional-characterization)
- [Human Behavioral Data](#human-behavioral-data)
  - [Experimental Design](#experimental-design)
  - [Stimulus Sets](#stimulus-sets)
  - [Data Loading](#data-loading)
- [The 2D Mechanistic Sandbox](#the-2d-mechanistic-sandbox)
- [Key Notebooks](#key-notebooks)
- [Tests](#tests)
- [Citation](#citation)
- [License](#license)

---

## Scientific Background

Sound is transient, so most auditory judgments implicate memory. Despite this, little is known about memory for complex, real-world sounds. This project addresses two interrelated questions:

1. **Psychophysics:** How does auditory recognition memory decay over time, and are some sounds systematically more memorable than others?
2. **Computational modeling:** Can a simple parametric model—where memory traces diffuse and drift toward a learned prior—account for both the aggregate forgetting curve and item-level effects?

Human participants performed a serial sound recognition task: they heard sequences of real-world sounds (environmental sounds, globalized music, auditory textures, or cross-cultural stimuli) and indicated whether each sound had been heard before. Recognition performance (measured as d') declines steeply over the first few intervening stimuli and then stabilizes—a pattern consistent across stimulus types. Item-level hit and false alarm rates are reliable across participants, and false alarms are predicted by perceptual similarity in representational space.

Cross-cultural experiments (US/UK online participants vs. Tsimane' and San Borja participants in Bolivia) reveal that between-group consistency in memorability decreases with cultural distance, suggesting that prior auditory experience shapes recognition memory.

---

## Model Overview

### Core Formulation

A memory trace $m(t) \in \mathbb{R}^D$ evolves as a stochastic differential equation (SDE) on an energy landscape defined by a learned prior $\pi(m)$:

$$dm(t) = \eta \, \nabla_m \log \pi(m(t)) \, dt + \sigma \, \text{diag}(\tilde{s}) \, dW(t)$$

where:
- $\nabla_m \log \pi(\cdot)$ is the **score function** of the prior (implemented by a pretrained neural network)
- $\eta > 0$ controls the strength of **deterministic drift** toward high-probability regions
- $\sigma > 0$ controls the magnitude of **stochastic diffusion**
- $\tilde{s} = s / s_{\text{rms}}$ is the normalized per-dimension standard deviation of the stimulus pool
- $W(t)$ is a standard Wiener process

Equivalently, defining the energy $U(m) = -\log \pi(m)$:

$$dm(t) = -\eta \, \nabla_m U(m(t)) \, dt + \sigma \, \text{diag}(\tilde{s}) \, dW(t)$$

Memory traces behave as particles undergoing stochastic relaxation in an energy landscape: drift pulls memories toward prototypical sounds (energy minima), while diffusion degrades them.

### The Two Main Models

The codebase implements two primary model families that share the same encoding and decision stages but differ in how stored traces evolve over time:

#### Model 1: Three-Step Noise Schedule (noise-only, no drift)

The simplest hypothesis: memory traces degrade through random diffusion only, with no systematic drift ($\eta = 0$). The noise magnitude depends on the trace's age via a **three-regime schedule**:

$$m_j^{(t+1)} = m_j^{(t)} + \sigma(a) \cdot \tilde{s} \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I_D)$$

where the noise magnitude $\sigma(a)$ is a step function of trace age $a$:

$$\sigma(a) = \begin{cases} \sigma_0 & \text{at encoding (applied once at insertion)} \\ \sigma_1 & \text{for } 1 < a < t_{\text{step}} \quad \text{(short delays, typically ISI 1–4)} \\ \sigma_2 & \text{for } a \geq t_{\text{step}} \quad \text{(long delays, typically ISI 8–64)} \end{cases}$$

| Parameter | Symbol | Role |
|-----------|--------|------|
| Encoding noise | σ₀ | Gaussian noise at memory insertion (applied once) |
| Short-delay noise | σ₁ | Per-step noise for young traces (age < t_step) |
| Long-delay noise | σ₂ | Per-step noise for old traces (age ≥ t_step) |
| Regime boundary | t_step | Age threshold between σ₁ and σ₂ regimes (default: 5) |

**Key finding:** A constant noise model cannot reproduce the shape of human forgetting curves. The three-regime schedule is required: steep early decay (large σ₁) followed by stabilization (small σ₂).

**Implementation:** `utls/runners_v2.py::run_model_core` with `ThreeRegimeNoise` schedule. Grid search: `src/model/run_3step_grid_search.py`.

#### Model 2: Prior-Guided Drift-Diffusion

Extends Model 1 with deterministic drift toward high-probability regions of a **learned prior** over natural sounds:

$$dm(t) = \eta \, \nabla_m \log \pi(m(t)) \, dt + \sigma \, \text{diag}(\tilde{s}) \, dW(t)$$

The score function $\nabla_m \log \pi(\cdot)$ is implemented by a pretrained score-based diffusion model (`ScoreNetAudioV2`) that was trained on auditory texture statistics. At each time step, memories are nudged toward prototypical sound representations. This model uses **constant** per-step noise (σ), not the three-regime schedule—the prior drift itself provides the mechanism for stabilization at long delays.

| Parameter | Symbol | Role |
|-----------|--------|------|
| Encoding noise | σ₀ | Gaussian noise at memory insertion (applied once) |
| Diffusion noise | σ | Constant per-step Langevin stochastic noise |
| Drift step size | η | Prior-driven drift magnitude per trial |

**Key prediction:** When $\eta = 0$ (no prior), forgetting is steep and monotonic. When $\eta$ is very large, all traces converge to the prior and discriminability collapses. An intermediate $\eta$ balances trace stability with distinctiveness. The stationary distribution is $p^*(m) = \pi(m)^{2\eta/\sigma^2}$; when $\sigma^2 = 2\eta$, traces converge exactly to the prior.

**Implementation:** `utls/runners_prior.py::run_model_core_prior` with `ScoreFunction`. Grid search: `src/model/run_prior_guided_grid_search.py`.

### Recognition Decision

At retrieval, the incoming probe $x_{\text{probe}}$ is compared to all stored memories via a distance metric (cosine distance by default):

$$d(x_{\text{probe}}, m_j) = 1 - \frac{x_{\text{probe}} \cdot m_j^{(t)}}{\|x_{\text{probe}}\| \, \|m_j^{(t)}\|}$$

The minimum distance across the memory bank serves as the decision variable. Scores are collected across trials into "hit" distributions (repeated stimuli) and "false alarm" distributions (novel stimuli), from which AUROC and d' are computed.

### Additional Model Variants

Beyond the two primary models, the codebase includes several alternative formulations explored during development:

| Model | Class | Description |
|-------|-------|-------------|
| **M1: Three-step noise** | `run_model_core` + `ThreeRegimeNoise` | Age-dependent noise schedule, no drift. Primary noise-only baseline. |
| **M2: Prior-guided drift** | `run_model_core_prior` + `ScoreFunction` | Langevin dynamics with learned prior. Primary drift model. |
| **Distance-based (OOP)** | `DistanceMemoryModel` | Object-oriented noise-only model for single-experiment runs. |
| **Prior-guided (OOP)** | `DistanceMemoryPriorModel` | Object-oriented prior-guided model with animation/visualization support. |
| **Likelihood-based** | `ApproximatePosteriorModel` | Recognition via log-likelihood under a Gaussian memory distribution. |
| **Power-law noise** | `DistanceMemoryPowerLawModel` | Noise variance decays as a power law of trace age. |
| **Scheduled noise** | `DistanceMemoryModelScheduledNoise` | Noise variance follows a configurable schedule. |
| **Mixture model** | `MixtureMemoryModel` / `NoisyAgeMixtureMemoryModel` | Mixture of Gaussians over memory traces with age-dependent noise. |

The functional simulation engines (`run_model_core`, `run_model_core_prior`) are preferred for grid searches and large-scale analyses. The OOP model classes (`DistanceMemoryModel`, `DistanceMemoryPriorModel`) provide richer introspection (trial-by-trial snapshots, memory bank animation) and are useful for single-experiment exploration and debugging.

---

## Repository Structure

```
memory/
├── src/model/                        # Core model implementations
│   ├── DistanceMemoryModel.py        # M1: noise-only baseline
│   ├── DistanceMemoryPriorModel.py   # M2: prior-guided drift-diffusion
│   ├── ScoreFunction.py              # Wrapper for the learned score network
│   ├── analytic_gmm_2d.py            # Analytic 2D GMM prior for sandbox
│   ├── score_adapter_2d.py           # ScoreFunction-compatible 2D wrapper
│   ├── ApproximatePosteriorModel.py  # Likelihood-based recognition
│   ├── DistanceMemoryPowerLawModel.py
│   ├── DistanceMemoryModelScheduledNoise.py
│   ├── MixtureMemoryModel.py
│   ├── NoisyAgeMixtureMemoryModel.py
│   ├── run_prior_guided_grid_search.py    # M2: Prior-guided 3D grid search (σ₀, σ, η)
│   ├── run_prior_guided_refined_pipeline.py
│   ├── run_3step_grid_search.py           # M1: Three-step noise 3D grid search (σ₀, σ₁, σ₂)
│   ├── run_2d_grid_search.py              # 2D sandbox grid search
│   ├── run_2d_grid_search_vectorized.py   # Vectorized 2D grid search
│   ├── main.py ... main_v6.py             # Entry points (versioned)
│   └── optimize_*.py                      # Fitting/optimization scripts
│
├── utls/                             # Utilities (simulation, analysis, plotting)
│   ├── runners_v2.py                 # M1 simulation engine (run_model_core) + noise schedules
│   ├── runners_prior.py              # M2 simulation engine (run_model_core_prior)
│   ├── runners_2d.py                 # 2D sandbox simulation engine
│   ├── runners_utils.py              # Data loading, encoder building, experiment orchestration
│   ├── sigma_fitting.py              # Sequential parameter fitting (3-stage)
│   ├── sigma_fitting_2d.py           # 2D sandbox parameter sweeps
│   ├── encoders.py                   # AudioTextureEncoder, PCA, DNN embeddings
│   ├── analysis_helpers.py           # d', ROC, cross-noise orchestration
│   ├── roc_utils.py                  # ROC curve and AUROC computation
│   ├── toy_experiments.py            # Sequence generation for parameter isolation
│   ├── human_analysis.py             # Human behavioral data processing
│   ├── human_plotting.py             # Human data visualization
│   ├── scaling.py                    # Noise scaling analysis
│   ├── drift_diagnostics.py          # Drift trajectory diagnostics
│   ├── data_loading.py               # Stimulus loading utilities
│   ├── analysis_2d.py                # 2D sandbox analysis
│   ├── sandbox_2d_data.py            # 2D grid stimuli + geometry
│   ├── train_prior_textures.py       # Prior training on texture statistics
│   └── prior_utls/                   # Prior-specific helpers (audio, normalization, projection)
│
├── utils/                            # General-purpose utilities
│   ├── sequence_utils.py             # ISISequence + StimulusManager for experiment design
│   ├── audio_utils.py                # Audio I/O and preprocessing
│   ├── cochleagram_utils.py          # Cochleagram computation
│   ├── distance_metrics.py           # Distance/similarity metrics
│   ├── dprime.py                     # d' computation utilities
│   ├── loading.py                    # Results loading with exclusion criteria
│   ├── reliability.py                # Split-half reliability analysis
│   ├── plot_utils.py / plotting.py   # Visualization helpers
│   └── pair_selection_utils.py       # Stimulus pair selection
│
├── notebooks/                        # Jupyter notebooks (chronological development log)
│   ├── human_analysis/               # Human behavioral data analyses
│   ├── 2d_guided_drift_sandbox.ipynb # End-to-end 2D sandbox demo
│   ├── example-usage-models.ipynb    # Model usage examples
│   └── ...                           # ~90 dated development notebooks
│
├── slurm-scripts/                    # HPC job submission
│   ├── run_3step_grid_search.sh          # M1 SLURM job script
│   ├── submit_3step_batches.py           # M1 batched submission
│   ├── submit_3step_driver.sh
│   ├── gather_3step_grid_search.sh
│   ├── run_prior_guided_grid_search.sh   # M2 SLURM job script (GPU)
│   ├── submit_prior_guided_batches.py    # M2 batched submission
│   ├── submit_prior_guided_driver.sh
│   ├── gather_prior_guided_grid_search.sh
│   ├── run_2d_grid_search*.sh
│   ├── gen_model_yamls*.py               # YAML config generators
│   └── ...
│
├── scripts/                          # Standalone scripts
│   ├── run_2d_grid_search.py
│   └── run_2d_grid_search.sh
│
├── tests/                            # Validation tests
│   └── test_2d_sandbox.py            # 6 tests for 2D sandbox correctness
│
├── reports/                          # Analysis summaries
│   └── 2d_guided_drift_summary.md
│
└── make_stability_notebooks.py       # Notebook generation for stability analyses
```

### File Activity Audit

The table below classifies every source file by its last git modification date (as of March 31, 2026). This is intended to guide cleanup for a public release — files marked **Legacy** are developmental artifacts that are not part of the current pipeline.

#### Active Core (last modified: March 17–31, 2026)

These files constitute the current working pipeline and should be retained in a cleaned repo.

| File | Last Modified | Role |
|------|--------------|------|
| `utls/runners_v2.py` | Mar 31 | M1 simulation engine + all noise schedules |
| `src/model/run_3step_grid_search.py` | Mar 30 | M1 grid search script |
| `src/model/run_prior_guided_grid_search.py` | Mar 28 | M2 grid search script |
| `src/model/ScoreFunction.py` | Mar 27 | Learned score function wrapper |
| `utls/runners_prior.py` | Mar 25 | M2 simulation engine |
| `src/model/run_prior_guided_refined_pipeline.py` | Mar 25 | M2 refined pipeline |
| `src/model/run_2d_grid_search_vectorized.py` | Mar 22 | 2D sandbox vectorized grid search |
| `utls/runners_utils.py` | Mar 20 | Data loading, encoder building, experiment orchestration |
| `utls/encoders.py` | Mar 20 | Audio texture encoder + PCA |
| `utls/runners_2d.py` | Mar 19 | 2D sandbox simulation engine |
| `src/model/run_2d_grid_search.py` | Mar 18 | 2D sandbox grid search |
| SLURM: `submit_prior_guided_batches.py`, `submit_3step_batches.py`, `submit_2d_vec_batches.py` + their `.sh` drivers and `gather_*.sh` scripts | Mar 18–29 | HPC job submission |

#### Supporting (last modified: February–mid-March 2026)

Active utility modules used by the core pipeline.

| File | Last Modified | Role |
|------|--------------|------|
| `utls/sigma_fitting.py` | Mar 13 | Sequential 3-stage parameter fitting (M1) |
| `utls/roc_utils.py` | Mar 11 | ROC/AUROC computation |
| `utls/drift_diagnostics.py` | Mar 10 | Drift trajectory analysis |
| `src/model/ApproximatePosteriorModel.py` | Mar 9 | Likelihood-based recognition model |
| `utls/toy_experiments.py` | Mar 7 | Toy sequence generation for parameter isolation |
| `utls/analysis_helpers.py` | Feb 27 | d' computation, cross-noise orchestration |
| `utls/prior_utls/*` | Feb 24 | Prior-specific helpers |
| `utls/scaling.py`, `human_plotting.py`, `human_analysis.py`, `data_loading.py` | Feb 20 | Human data analysis utilities |
| 2D sandbox: `analytic_gmm_2d.py`, `score_adapter_2d.py`, `sandbox_2d_data.py`, `analysis_2d.py`, `sigma_fitting_2d.py` | Mar 15 | Fully self-contained 2D sandbox |
| `tests/test_2d_sandbox.py` | Mar 15 | Sandbox validation tests |

#### Legacy (last modified: March 18 bulk commit or earlier — not in current pipeline)

These files were committed in a bulk addition on March 18 but reflect earlier development stages. They are **not called by the active pipeline** and are candidates for removal.

| File | Notes |
|------|-------|
| `src/model/DistanceMemoryModel.py` | OOP noise-only model; replaced by `run_model_core` |
| `src/model/DistanceMemoryPriorModel.py` | OOP prior-guided model; replaced by `run_model_core_prior` |
| `src/model/MixtureMemoryModel.py` | Mixture-of-Gaussians variant (exploratory) |
| `src/model/NoisyAgeMixtureMemoryModel.py` | Age-dependent mixture variant (exploratory) |
| `src/model/DistanceMemoryPowerLawModel.py` | Power-law noise variant (exploratory) |
| `src/model/DistanceMemoryModelScheduledNoise.py` | Scheduled noise variant (Feb 17; superseded by `ThreeRegimeNoise`) |
| `src/model/main.py` through `main_v6.py`, `main-singleisi.py` | Versioned entry points (superseded by grid search scripts) |
| `src/model/optimize_*.py` (3 files) | Early optimization scripts (superseded by `sigma_fitting.py` + grid searches) |
| `src/model/subsample_analysis.py` | Subsample convergence analysis |
| `make_stability_notebooks.py` | Notebook auto-generation for stability sweeps |
| Old SLURM: `run_model_fit*.sh`, `run_model_grid*.sh`, `run_model_yamls*.sh`, `run_generate-*.sh`, `run_memory_models.sh`, `run_score_analysis.sh`, `run_sigma*_stability.sh`, `run_three_stage_*.sh`, `run_subsample_analysis.sh`, `run_coarse_param_optimization.sh`, `test_v5_pipeline.sh`, `gen_model_yamls*.py` | Earlier SLURM infrastructure |

#### Stale (last modified: June–August 2025)

The entire `utils/` directory (distinct from `utls/`) has not been touched since mid-2025. Some modules (e.g., `sequence_utils.py`, `loading.py`) may still be imported transitively, but most are candidates for consolidation into `utls/` or removal.

| File | Last Modified |
|------|--------------|
| `utils/sequence_utils.py` | Jul 2025 |
| `utils/loading.py`, `utils/reliability.py`, `utils/dprime.py`, `utils/plotting.py` | Aug 2025 |
| `utils/audio_utils.py` | Jul 2025 |
| `utils/cochleagram_utils.py`, `utils/cochleagram_params.yaml` | Jun 2025 |
| `utils/distance_metrics.py`, `utils/stationarity.py`, `utils/plot_utils.py`, `utils/pair_selection_utils.py`, `utils/extreme_pairs_utils.py` | Jun 2025 |

---

## Installation

### Dependencies

Core Python dependencies (Python 3.10+):

```
torch >= 2.0
numpy
scipy
scikit-learn
pandas
matplotlib
seaborn
pyyaml
tqdm
```

### External Libraries

The full model (not the 2D sandbox) depends on several lab-internal libraries that must be available on `sys.path`:

| Library | Purpose | Expected Location |
|---------|---------|-------------------|
| `chexture_choolbox` | Auditory texture model (McDermott & Simoncelli statistics) | `/orcd/data/jhm/001/om2/jmhicks/projects/TextureStreaming/code/` |
| `texture_prior` | Texture prior parameters, normalization, dataset config | Same as above |
| `audio-prior` | Score-based diffusion model (`ScoreNetAudioV2`) + SDE utilities | `/orcd/data/jhm/001/om2/lakshmin/audio-prior/` |

These paths are currently hardcoded and assume MIT/BCS ORCD cluster access. For external use, update the `sys.path.append(...)` calls in `ScoreFunction.py`, `utls/encoders.py`, and `utls/runners_utils.py`.

### Score Model Configuration

The learned prior (score function) is configured via a YAML file (default: `bryan.yaml`):

```yaml
train:
  n_epochs    : 512
  batch_size  : 4096
  lr          : 0.0001
data:
  data_root       : '/path/to/texture_statistics_4096texturePCs.pt'
  data_mix_root   : '/path/to/mixture_statistics_4096PCs.pt'
  n_pcs       : 256
  var_scale   : False
model:
  ckpt_path   : 'ckpts/texture_diffusion2D_prior_{}pcs_mode_{}.pth'
  use_single_dim_conv : False
  embed_dim   : 256
  channels    : [32, 64, 128]
  kernel_size : 5
  dilations   : [1, 2, 4, 8, 16]
sample:
  sample_batch_size : 4096
  num_steps   : 2500
```

Key parameters: the prior operates in a **256-PC** subspace of auditory texture statistics (McDermott & Simoncelli, 2011). The score network (`ScoreNetAudioV2`) is a 1D convolutional architecture with dilated convolutions and sinusoidal time embeddings, trained via denoising score matching on ~4096 natural texture recordings. Checkpoint paths are resolved relative to the `audio-prior` repository.

The **2D sandbox** (`analytic_gmm_2d.py`, `runners_2d.py`, `sandbox_2d_data.py`) has no external dependencies beyond PyTorch and NumPy—use it to explore the model dynamics without cluster access.

---

## Quickstart

### 2D Sandbox (No External Dependencies)

The 2D sandbox replaces the learned high-dimensional prior with an analytic 3-component Gaussian mixture model in ℝ². It is fully self-contained and useful for understanding model dynamics, parameter sensitivity, and ablation experiments.

```python
import torch
from src.model.analytic_gmm_2d import make_default_gmm
from src.model.score_adapter_2d import ScoreAdapter2D
from utls.sandbox_2d_data import make_2d_grid_stimuli
from utls.runners_2d import run_2d_isi_sweep

# Set up analytic GMM prior and adapter
gmm = make_default_gmm()
score_model = ScoreAdapter2D(gmm, normalize=True)

# Generate 80 grid stimuli in [-4, 4]²
X0, name_to_idx, _ = make_2d_grid_stimuli()

# Sweep ISIs with fixed parameters
isi_values = [0, 1, 2, 4, 8, 16, 32]
results = run_2d_isi_sweep(
    X0=X0, name_to_idx=name_to_idx,
    score_model=score_model,
    sigma0=0.5, sigma=0.1, eta=0.02,
    isi_values=isi_values, n_mc=20,
)
# results contains d' per ISI
```

See `notebooks/2d_guided_drift_sandbox.ipynb` for the full pipeline with visualization.

### Full Model with Auditory Textures

Requires cluster access and external libraries (see [Installation](#external-libraries)).

#### M1: Three-Step Noise Schedule (no drift)

```python
from utls.runners_v2 import run_model_core, make_noise_schedule
from utls.runners_utils import load_experiment_data, build_encoder, encode_stimuli

# Load human experiment data
experiment_list, all_files, name_to_idx, human_runs = load_experiment_data(
    which_task=2, which_isi=16, is_multi=False,
)

# Build encoder and encode stimuli into 256-PC texture space
encoder = build_encoder(encoder_type='texture_pca', pc_dims=256, device='cuda')
X0 = encode_stimuli(all_files, encoder, name_to_idx)

# Create three-regime noise schedule
noise_schedule = make_noise_schedule('three-regime', {
    'sigma0': 5.0,     # encoding noise
    'sigma1': 2.0,     # short-delay noise (age < t_step)
    'sigma2': 0.1,     # long-delay noise (age >= t_step)
    't_step': 5,       # regime boundary
})

# Run noise-only simulation
run = run_model_core(
    sigma0=5.0, X0=X0, name_to_idx=name_to_idx,
    experiment_list=experiment_list,
    noise_schedule=noise_schedule,
    metric='cosine',
)
```

#### M2: Prior-Guided Drift-Diffusion

```python
from utls.runners_prior import run_model_core_prior
from src.model.ScoreFunction import ScoreFunction

# (X0, name_to_idx, experiment_list loaded as above)

# Load learned score function (prior)
score_model = ScoreFunction(device='cuda', normalize=True)

# Run prior-guided simulation
run = run_model_core_prior(
    sigma0=0.1, sigma=0.05, drift_step_size=0.02,
    X0=X0, name_to_idx=name_to_idx,
    experiment_list=experiment_list,
    score_model=score_model,
    metric='cosine',
)
```

---

## Usage

### Running a Single Simulation

The two main simulation engines are:

- **`utls/runners_v2.py::run_model_core`** — M1 (noise-only). Takes a `NoiseSchedule` object that controls age-dependent noise. No score model needed. CPU-only.
- **`utls/runners_prior.py::run_model_core_prior`** — M2 (prior-guided). Takes a score model with a `.forward(x)` method. Requires GPU for the score network.

Both accept stimulus embeddings `X0` (any dimensionality), experiment sequences (`experiment_list`: list of lists of stimulus names), and return per-ISI hit/FA score distributions for downstream ROC/d' computation.

### Grid Search over Parameters

Both models support exhaustive 3D grid searches over their respective parameter spaces:

#### M1: Three-Step Noise Grid Search

Sweeps (σ₀, σ₁, σ₂) with a fixed `t_step` (default: 5). Uses `run_model_core` + `ThreeRegimeNoise`.

```bash
# Single triple (local)
python src/model/run_3step_grid_search.py \
    --job-index 0 --n-mc 10 --metric cosine --which-task 2

# Merge results after completion
python src/model/run_3step_grid_search.py \
    --merge --save-dir reports/figures/3step_grid_search_t5
```

Default grid: 15 × 15 × 15 = 3,375 parameter triples (log-spaced from 0.01 to 25, plus 0.0). Special cases within this grid: σ₀ = 0 is the no-encoding-noise model; σ₁ = σ₂ reduces to constant noise; σ₂ = 0 eliminates long-delay noise entirely.

#### M2: Prior-Guided Grid Search

Sweeps (σ₀, σ, η). Uses `run_model_core_prior` + `ScoreFunction`.

```bash
# Single triple (local)
python src/model/run_prior_guided_grid_search.py \
    --job-index 0 --n-mc 10 --metric euclidean --which-task 2

# Merge results after completion
python src/model/run_prior_guided_grid_search.py \
    --merge --save-dir reports/figures/prior_guided_grid_search
```

Default grid: 13 × 13 × 13 = 2,197 parameter triples (linearly spaced from 0.01 to 1.0, plus 0.0).

Both grid searches run `n_mc` Monte Carlo repetitions of the full experiment per parameter triple to obtain stable d' estimates.

### SLURM Cluster Execution

For large-scale grid searches, the repo provides batched SLURM submission pipelines for both models:

```bash
# M1: Three-step noise grid search
python slurm-scripts/submit_3step_batches.py
python slurm-scripts/submit_3step_batches.py --dry-run

# M2: Prior-guided grid search
python slurm-scripts/submit_prior_guided_batches.py
python slurm-scripts/submit_prior_guided_batches.py --dry-run

# Override parameters (same interface for both)
python slurm-scripts/submit_prior_guided_batches.py \
    --metric euclidean --n-mc 5 --which-task 2

# Auto-merge after all batches
python slurm-scripts/submit_prior_guided_batches.py --gather
```

The submission scripts handle SLURM's `QOSMaxSubmitJobPerUserLimit` by polling `squeue` and only submitting the next batch (150 array tasks) after the current one completes. The M2 prior-guided search requires GPU allocation (`--gres=gpu:1`) for score function evaluation.

### Merging Grid Search Results

After all SLURM jobs complete, merge per-slice `.npz` files into a single results array:

```bash
# M1: Three-step
python src/model/run_3step_grid_search.py \
    --merge --save-dir /path/to/3step_results/

# M2: Prior-guided
python src/model/run_prior_guided_grid_search.py \
    --merge --save-dir /path/to/prior_results/
```

Output: `grid_search_results_3step_t5.npz` or `grid_search_results_prior_guided.npz`, containing d' arrays indexed by the respective 3D parameter grid × ISI.

### Parameter Fitting

The `utls/sigma_fitting.py` module implements a sequential three-stage fitting procedure for **M1** (the noise-only model). By constructing toy experiments where all repeats occur at a specific ISI, each noise parameter can be isolated:

1. **Stage 1 (σ₀):** Fit using ISI=0 experiments. Immediate repeats isolate encoding noise—no drift noise has been applied yet.
2. **Stage 2 (σ₁):** Fix σ₀, fit using ISI 1–4 experiments. Short-delay repeats are sensitive to the early noise regime.
3. **Stage 3 (σ₂):** Fix σ₀ and σ₁, fit using ISI 8–64 experiments. Long-delay repeats probe the asymptotic noise regime.

Each stage performs a 1D grid search with iterative refinement (log-spaced or hybrid grids), avoiding expensive multi-dimensional optimization. The fitting target is MSE between model and human d' at each ISI.

---

## Analysis Pipeline

### Computing d' from Simulations

The simulation engine returns raw hit and false alarm score distributions. The conversion to d' follows:

1. Compute ROC curves per ISI via `utls/roc_utils.py::roc_from_arrays`
2. Extract AUROC
3. Convert to d' via $d' = \sqrt{2} \, \Phi^{-1}(\text{AUROC})$

```python
from utls.roc_utils import roc_from_arrays
from utls.analysis_helpers import auroc_to_dprime

fpr, tpr, auroc = roc_from_arrays(hit_scores, fa_scores, score_type='distance')
dprime = auroc_to_dprime(auroc)
```

### Human–Model Comparison

Human behavioral data is loaded via `utls/runners_utils.py::load_experiment_data`, which handles participant exclusion (min d' ≥ 2, min 120 trials), sequence deduplication, and ISI extraction. Model d' curves are compared to human d' curves via MSE or cosine similarity.

Item-level analyses compare per-stimulus model scores to human hit rates and false alarm rates, testing whether the model captures which sounds are consistently memorable or confusable.

### Grid Search Functional Characterization

Each d'(ISI) curve from the grid search is fit to an exponential decay:

$$d'(\text{ISI}) = A \, e^{-\lambda \cdot \text{ISI}} + C$$

where $A$ is the amplitude (peak-to-floor drop), $\lambda$ is the decay rate (forgetting speed), and $C$ is the asymptotic floor. Curves with std(d') < 0.05 are classified as flat. The resulting functional parameters are visualized as heatmaps over the (σ, η) grid for representative σ₀ slices.

---

## Human Behavioral Data

### Experimental Design

Participants hear sequences of ~120 sounds (2s each, 0.5s ISI). Half of the sounds are presented twice. After each sound, participants report whether they have heard it before. The **interstimulus interval** (ISI) is defined as the number of intervening stimuli between a sound's first and second presentation.

Two experimental designs are used:

- **Single-ISI experiments:** All repeats within a sequence share the same ISI (e.g., ISI=16). Useful for isolating item-level effects.
- **Multi-ISI experiments:** Repeats span multiple ISIs within one sequence (e.g., ISI ∈ {0, 1, 2, 4, 8, 16, 32, 64}). Used to characterize the full forgetting curve.

### Stimulus Sets

| Index | Set | Description |
|-------|-----|-------------|
| 0 | Environmental sounds | Industrial and nature recordings |
| 1 | Globalized music | Music clips from diverse cultures |
| 2 | Auditory textures | Stationary "background" sounds (rain, fire, machinery) |
| 3 | Natural History of Song | Cross-cultural vocal recordings |

### Data Loading

```python
from utls.runners_utils import load_experiment_data

experiment_list, all_files, name_to_idx, human_runs = load_experiment_data(
    which_task=2,       # stimulus set index
    which_isi=16,       # ISI condition (for single-ISI)
    is_multi=True,      # True for multi-ISI experiments
)
```

Human data paths assume ORCD cluster access. The loader applies exclusion criteria and returns experiment sequences aligned with the model's expected input format.

---

## The 2D Mechanistic Sandbox

A self-contained environment for exploring the prior-guided model with full analytical control. Replaces the learned 256-PC prior with a 3-component Gaussian mixture in ℝ², where density, score, and local geometry are all available in closed form.

**GMM Configuration:**

| Component | Mean | Covariance | Weight |
|-----------|------|------------|--------|
| Broad | [-2.5, 0.0] | diag(1.5, 1.5) | 0.40 |
| Tight-1 | [2.0, 2.0] | diag(0.4, 0.4) | 0.30 |
| Tight-2 | [2.0, -2.0] | diag(0.4, 0.4) | 0.30 |

Stimuli are 80 grid points on a regular 9×9 lattice in [-4, 4]² (not sampled from the GMM). The sandbox supports matched vs. mismatched prior ablations via `make_mismatched_gmm()`.

**Key files:** `src/model/analytic_gmm_2d.py`, `src/model/score_adapter_2d.py`, `utls/sandbox_2d_data.py`, `utls/runners_2d.py`, `utls/analysis_2d.py`.

See `reports/2d_guided_drift_summary.md` for a detailed summary of sandbox findings.

---

## Key Notebooks

The `notebooks/` directory contains a chronological development log (~90 notebooks). Key entry points:

| Notebook | Purpose |
|----------|---------|
| `2d_guided_drift_sandbox.ipynb` | End-to-end 2D sandbox demo with visualizations |
| `example-usage-models.ipynb` | Model instantiation and basic usage |
| `human_analysis/01_multi_isi_analysis.ipynb` | Human forgetting curve analysis (multi-ISI) |
| `human_analysis/02_universality_analysis.ipynb` | Cross-stimulus-set universality |
| `human_analysis/03_itemwise_analysis.ipynb` | Item-level hit/FA reliability |
| `2026-03-22_3step-grid-search-model-comparison.ipynb` | M1: Three-step grid search results and model comparison |
| `2026-03-27_prior-grid-search-model-comparison.ipynb` | M2: Prior-guided grid search results |
| `2026-03-30_3step-nelder-mead-optimization.ipynb` | Nelder-Mead parameter optimization |
| `2026-02-10_score-function-revisited.ipynb` | Score function analysis and validation |
| `sequential_sigma_fitting.ipynb` | Three-stage sequential fitting procedure (M1) |
| `2026-03-17_grid-search-analysis.ipynb` | Grid search functional characterization (exponential decay fits) |

---

## Tests

Validation tests for the 2D sandbox verify:

1. Analytic score matches finite-difference $\nabla \log p(x)$
2. Component posteriors sum to 1
3. Score adapter produces correct output shapes
4. Simulation engine runs without error and returns expected keys
5. Reproducibility under fixed seeds
6. d' values are finite across ISIs

```bash
python -m pytest tests/test_2d_sandbox.py -v
```

---

## Citation

If you use this code, please cite:

```
Medina, B. J., Govindarajan, L. N., Fiete, I. R., & McDermott, J. H.
A computational model of auditory memory recognition with prior-guided memory traces.
(In preparation)
```

and the companion psychophysics paper:

```
Medina, B. J. & McDermott, J. H.
The psychophysics of auditory memory for real-world sounds.
(In preparation)
```

---

## License

TBD
