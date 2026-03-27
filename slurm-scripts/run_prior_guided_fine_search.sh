#!/bin/bash
#SBATCH -J refined_grid
#SBATCH -p ou_bcs_normal
#SBATCH -t 0-0:15:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --array=0-149
#SBATCH -o slurm-scripts/logs/%x_%A_%a.out
#SBATCH -e slurm-scripts/logs/%x_%A_%a.out

conda activate /orcd/data/jhm/001/bjmedina/miniconda3/envs/asr_312_312
cd /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory || exit 1

OFFSET=0
BATCH_SIZE=150
JOB_INDEX=$(( OFFSET * BATCH_SIZE + SLURM_ARRAY_TASK_ID ))

MODEL_TYPE="3step"   # prior | 3step
PARALLEL_MODE="flat"
N_MC=3
METRIC="cosine"
T_STEP=4
SAVE_DIR="${SAVE_DIR:-/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/reports/figures/refined_pipeline_${MODEL_TYPE}}"
COARSE_RESULTS_PATH="/orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/reports/figures/3step_grid_search_metric-${METRIC}_t${T_STEP}_nmc1_task0"

python /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory/src/model/run_prior_guided_refined_pipeline.py \
  --mode fine-search \
  --model-type "$MODEL_TYPE" \
  --job-index "$JOB_INDEX" \
  --parallel-mode "$PARALLEL_MODE" \
  --n-mc "$N_MC" \
  --metric "$METRIC" \
  --t-step "$T_STEP" \
  --save-dir "$SAVE_DIR" \
  --coarse-results-path "$COARSE_RESULTS_PATH" \
  --resume
