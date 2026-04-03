#!/usr/bin/env python3
"""
Generate one YAML per run.
Each YAML corresponds to exactly one Slurm job.
"""

from pathlib import Path
import yaml
import itertools


OUT_DIR = Path("yamls")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def base_config():
    """Return base config shared by all runs."""
    return {
        "results_root": "/om2/user/bjmedina/auditory-memory/memory/figures/model_results_v2",
        "tag": "slurm",
        "experiment": {
            "is_multi": True,
            "n_seqs": -1,
            "n_samples": 50,
        },
    }


# ---------------------------
# GRIDS (explicit & safe)
# ---------------------------

tasks = {
    0: "ind-nature",
    1: "global-music",
    2: "atexts",
}

metrics = ["cosine", "euclidean"]

noise_models = [
    {
        "name": "two-regime",
        "sigma0_min": 0.0,
        "sigma0_max": 2.0,
        "sigma1_min": 0.0,
        "sigma1_max": 1.0,
        "t_step": 3,
    },
    {
        "name": "two-regime",
        "sigma0_min": 0.0,
        "sigma0_max": 2.0,
        "sigma1_min": 0.0,
        "sigma1_max": 1.0,
        "t_step": 4,
    },
    {
        "name": "two-regime",
        "sigma0_min": 0.0,
        "sigma0_max": 2.0,
        "sigma1_min": 0.0,
        "sigma1_max": 1.0,
        "t_step": 5,
    },
    {
        "name": "const",
        "sigma0_min": 0.0,
        "sigma0_max": 2.0,
    },
]

representations = [
    ("kell2018", {"layer": ["input_after_preproc", "relu1", "relu2", "relu3", "relu4", "relufc"]}),
    ("resnet50", {"layer": ["input_after_preproc", "layer1", "layer2", "layer3", "layer4"]}),
    ("texture", {"pc_dims": [128, 256]}),
]

# ---------------------------
# GENERATION
# ---------------------------

run_idx = 0

for task_id, task_name in tasks.items():
    for metric in metrics:
        for noise in noise_models:
            for repr_type, repr_params in representations:

                cfg_base = base_config()
                cfg_base["experiment"]["which_task"] = task_id
                cfg_base["metric"] = metric
                cfg_base["noise_model"] = noise

                # -------- Layer-based representations --------
                if "layer" in repr_params:
                    for layer in repr_params["layer"]:
                        cfg = cfg_base.copy()
                        cfg["run_id"] = f"run_{run_idx:06d}"
                        cfg["representation"] = {
                            "type": repr_type,
                            "layer": layer,
                        }

                        out = OUT_DIR / f"{cfg['run_id']}.yaml"
                        with open(out, "w") as f:
                            yaml.safe_dump(cfg, f, sort_keys=False)

                        run_idx += 1

                # -------- PC-based representations --------
                if "pc_dims" in repr_params:
                    for pc in repr_params["pc_dims"]:
                        cfg = cfg_base.copy()
                        cfg["run_id"] = f"run_{run_idx:06d}"
                        cfg["representation"] = {
                            "type": repr_type,
                            "pc_dims": pc,
                        }

                        out = OUT_DIR / f"{cfg['run_id']}.yaml"
                        with open(out, "w") as f:
                            yaml.safe_dump(cfg, f, sort_keys=False)

                        run_idx += 1
                        
print(f"Generated {run_idx} YAML files")