#!/bin/bash
#SBATCH -J mem_models
#SBATCH -o /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/logs/%x_%j.out
#SBATCH -e /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/logs/%x_%j.err
#SBATCH -p normal
#SBATCH -t 24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=bjmedina@mit.edu

# ==============================
# Environment setup
# ==============================
module load anaconda
source activate your_env_name   # replace with your conda env name

# ==============================
# Paths
# ==============================
BASE_DIR="/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory"
SCRIPT="${BASE_DIR}/src/main.py"

X0_PATH="${BASE_DIR}/data/X0_tensor.pt"
IDX_PATH="${BASE_DIR}/data/name_to_idx.pt"
EXPS_PATH="${BASE_DIR}/data/experiment_list.pt"
SAVE_BASE="${BASE_DIR}/figures/model-results/"

# ==============================
# Model configuration
# ==============================
noise_modes=("power-decay" "exp-decay")
metrics=("euclidean" "likelihood")
encoder="texture_statistics"
sigmas=(0.1 0.2 0.3 0.4 0.5)

# ==============================
# Loop through configurations
# ==============================
for nm in "${noise_modes[@]}"; do
  for met in "${metrics[@]}"; do

    if [[ "$met" == "likelihood" ]]; then
      model="LikelihoodMemoryModel"
    else
      model="DistanceMemoryModel"
    fi

    for sigma0 in "${sigmas[@]}"; do
      run_id="${model}_${met}_${nm}_sigma${sigma0}_$(date +%Y%m%d_%H%M%S)"
      echo ">>> Running $model | metric=$met | noise_mode=$nm | sigma0=$sigma0"

      python $SCRIPT \
        --metric "$met" \
        --noise_mode "$nm" \
        --encoder "$encoder" \
        --sigma0 "$sigma0" \
        --run_id "$run_id" \
        --save_base "$SAVE_BASE" \
        --X0_path "$X0_PATH" \
        --idx_path "$IDX_PATH" \
        --exp_list_path "$EXPS_PATH"

      echo ">>> Finished $model | metric=$met | noise_mode=$nm | sigma0=$sigma0"
      echo "-------------------------------------------------------------"
    done
  done
done
