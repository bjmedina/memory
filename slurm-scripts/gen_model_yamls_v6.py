#!/usr/bin/env python3
"""
Generate one YAML per (task, metric, representation, time_avg) combo
for compact three-stage sigma fitting via main_v6.py.

Usage:
    python gen_model_yamls_v6.py --out-dir ../model_yamls/v15_three-stage-compact
"""

from pathlib import Path
import copy
import yaml
import argparse


def base_config(n_samples=50, n_seqs=36):
    return {
        "results_root": "/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory",
        "tag": "slurm",
        "experiment": {
            "is_multi": True,
            "n_seqs": n_seqs,
            "n_samples": n_samples,
        },
    }


def three_regime_noise(
    sigma0_min=1.0, sigma0_max=40.0,
    sigma1_min=0.1, sigma1_max=40.0,
    sigma2_min=0.0005, sigma2_max=20.0,
    t_step=5,
):
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
    return {
        "n_grid": 15,
        "n_mc": 32,
        "n_refine_iters": 3,
        "n_experiments_per_isi": 20,
        "k_stimuli_per_exp": 5,
    }


def compact_fitting_defaults():
    return {
        "sigma1_isis": [1, 2, 4],
        "sigma1_length": 60,
        "sigma1_n_seqs": 30,
        "sigma1_min_pairs": 5,
        "sigma2_isis": [8, 16, 32, 64],
        "sigma2_length": 75,
        "sigma2_n_seqs": 30,
        "sigma2_min_pairs": 5,
        "n_seqs_per_rep": 10,
    }


TASKS = {
    0: "ind-nature",
    1: "global-music",
    2: "atexts",
}

METRICS = ["cosine"]
NOISE_MODELS = [three_regime_noise()]

REPRESENTATIONS = [
    ("resnet50", {"layer": ["layer3", "layer4", "avgpool"]}),
    ("kell2018", {"layer": ["relu4", "relufc"]}),
]

TIME_AVG_OPTIONS = [True, False]


def write_yaml(cfg, out_dir, run_idx):
    out = out_dir / f"run_{run_idx:06d}.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(
        description="Generate YAML configs for compact three-stage fitting via main_v6.py"
    )
    parser.add_argument(
        "--out-dir", type=str,
        default="../model_yamls/v15_three-stage-compact",
        help="Output directory for YAML files",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_idx = 0

    for task_id, _ in TASKS.items():
        for metric in METRICS:
            for noise in NOISE_MODELS:
                for repr_type, repr_params in REPRESENTATIONS:
                    if repr_type == "texture_pca" and task_id in (0, 1):
                        continue

                    for time_avg in TIME_AVG_OPTIONS:
                        if repr_type == "texture_pca" and not time_avg:
                            continue

                        cfg = base_config()
                        cfg["experiment"]["which_task"] = task_id
                        cfg["metric"] = metric
                        cfg["noise_model"] = copy.deepcopy(noise)
                        cfg["fitting"] = fitting_defaults()
                        cfg["compact_fitting"] = compact_fitting_defaults()

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
