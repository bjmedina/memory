#!/usr/bin/env python3
"""
Generate one YAML per (task, metric, representation, time_avg) combo
for three-stage sigma fitting via main_v5.py.

Usage:
    python gen_model_yamls_v5.py --out-dir ../model_yamls/v13_three-stage

Each generated YAML includes a ``fitting`` section with the three_stage_fit
hyperparameters (n_grid, n_mc, etc.) so they are tracked per-run.
"""

from pathlib import Path
import copy
import yaml
import argparse


# ── config helpers ────────────────────────────────────────────────────

def base_config(n_samples=50, n_seqs=36):
    """Return base config shared by all runs."""
    return {
        "results_root": "/om2/user/bjmedina/auditory-memory/memory",
        "tag": "slurm",
        "experiment": {
            "is_multi": True,
            "n_seqs": n_seqs,
            "n_samples": n_samples,
        },
    }


def three_regime_noise(
    sigma0_min=0.7, sigma0_max=0.75,
    sigma1_min=0.1, sigma1_max=0.6,
    sigma2_min=0.0005, sigma2_max=0.1,
    t_step=5,
):
    """Default three-regime noise config."""
    return {
        "name": "three-regime",
        "sigma0_min": sigma0_min,
        "sigma0_max": sigma0_max,
        "sigma1_min": sigma1_min,
        "sigma1_max": sigma1_max,
        "sigma2_min": sigma2_min,
        "sigma2_max": sigma2_max,
        "t_step": t_step,
    }


def fitting_defaults():
    """Default three_stage_fit hyperparameters."""
    return {
        "n_grid": 15,
        "n_mc": 32,
        "n_refine_iters": 2,
        "n_experiments_per_isi": 20,
        "k_stimuli_per_exp": 10,
    }


# ── grids ─────────────────────────────────────────────────────────────

TASKS = {
    0: "ind-nature",
    1: "global-music",
    2: "atexts",
}

METRICS = ["cosine"]

NOISE_MODELS = [three_regime_noise()]

REPRESENTATIONS = [
    # resnet50 layers
    ("resnet50", {"layer": [
        "input_after_preproc", "conv1_relu1",
        "layer1", "layer2", "layer3", "layer4", "avgpool",
    ]}),
    # kell2018 layers
    ("kell2018", {"layer": [
        "relu0", "relu1", "relu2", "relu3", "relu4", "relufc",
    ]}),
    # texture PCA dimensions (only for atexts)
    ("texture_pca", {"pc_dims": [64, 128, 256, 512, 1028]}),
]

TIME_AVG_OPTIONS = [True, False]


# ── generation ────────────────────────────────────────────────────────

def write_yaml(cfg, out_dir, run_idx):
    out = out_dir / f"run_{run_idx:06d}.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(
        description="Generate YAML configs for three-stage fitting via main_v5.py"
    )
    parser.add_argument(
        "--out-dir", type=str,
        default="../model_yamls/v13_three-stage",
        help="Output directory for YAML files (default: ../model_yamls/v13_three-stage)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_idx = 0

    for task_id, task_name in TASKS.items():
        for metric in METRICS:
            for noise in NOISE_MODELS:
                for repr_type, repr_params in REPRESENTATIONS:

                    # texture_pca only applies to atexts (task 2)
                    if repr_type == "texture_pca" and task_id in (0, 1):
                        continue

                    for time_avg in TIME_AVG_OPTIONS:
                        # texture_pca does not have a time_avg variant
                        if repr_type == "texture_pca" and not time_avg:
                            continue

                        cfg = base_config()
                        cfg["experiment"]["which_task"] = task_id
                        cfg["metric"] = metric
                        cfg["noise_model"] = copy.deepcopy(noise)
                        cfg["fitting"] = fitting_defaults()

                        # ---- layer-based representations ----
                        if "layer" in repr_params:
                            for layer in repr_params["layer"]:
                                cfg_copy = copy.deepcopy(cfg)
                                cfg_copy["run_id"] = f"run_{run_idx:06d}"
                                cfg_copy["representation"] = {
                                    "type": repr_type,
                                    "layer": layer,
                                    "time_avg": time_avg,
                                }
                                write_yaml(cfg_copy, out_dir, run_idx)
                                run_idx += 1

                        # ---- PC-based representations ----
                        if "pc_dims" in repr_params:
                            for pc in repr_params["pc_dims"]:
                                cfg_copy = copy.deepcopy(cfg)
                                cfg_copy["run_id"] = f"run_{run_idx:06d}"
                                cfg_copy["representation"] = {
                                    "type": repr_type,
                                    "pc_dims": pc,
                                }
                                write_yaml(cfg_copy, out_dir, run_idx)
                                run_idx += 1

    print(f"Generated {run_idx} YAML files in {out_dir}")
    print(f"Set SLURM --array=0-{run_idx - 1}")


if __name__ == "__main__":
    main()
