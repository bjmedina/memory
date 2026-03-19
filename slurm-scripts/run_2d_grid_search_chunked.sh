#!/bin/bash
#SBATCH -J 2d_grid_chunked
#SBATCH -p mit_normal_gpu
#SBATCH -t 0-4:00:00
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --mem=4G
## Chunked mode: each array job processes a chunk of flat-index triples.
## Default: 50 array jobs, each handling ~(TOTAL_TRIPLES / 50) triples.
## Adjust --array and N_JOBS together (they must match).
#SBATCH --array=0-49
#SBATCH -o slurm-scripts/logs/%x_%A_%a.out
#SBATCH -e slurm-scripts/logs/%x_%A_%a.err

# =============================
# ENVIRONMENT SETUP
# =============================
source activate /orcd/data/jhm/001/gelbanna/miniconda3/envs/asr_312
cd /orcd/data/jhm/001/om2/bjmedina/auditory-memory/memory || exit 1

# =============================
# CONFIGURABLE PARAMETERS
# (override via env vars when submitting)
#
# Examples:
#   # Default 20x20x20 grid, 50 jobs:
#   SIGMA0_GRID="0.0 0.05 ... 1.0" SIGMA_GRID="..." ETA_GRID="..." \
#     sbatch slurm-scripts/run_2d_grid_search_chunked.sh
#
#   # Custom number of jobs (must match --array):
#   N_JOBS=25 sbatch --array=0-24 slurm-scripts/run_2d_grid_search_chunked.sh
#
#   # After all jobs finish, merge:
#   python src/model/run_2d_grid_search_vectorized.py --merge --save-dir <SAVE_DIR>
# =============================

N_MC=${N_MC:-10}
ISIS="${ISIS:-0 2 8 16}"
FINE_GRID="${FINE_GRID:-true}"
SAVE_DIR="${SAVE_DIR:-reports/figures/2d_grid_search_vectorized_full}"
SEED="${SEED:-43}"
METRIC="${METRIC:-euclidean}"
N_SEQUENCES="${N_SEQUENCES:-100}"
SEQ_LENGTH="${SEQ_LENGTH:-99}"
MIN_PAIRS="${MIN_PAIRS:-5}"

# Number of array jobs (must match #SBATCH --array count)
N_JOBS="${N_JOBS:-50}"

# Custom grids (optional; override --fine when set).
SIGMA0_GRID="${SIGMA0_GRID:-}"
SIGMA_GRID="${SIGMA_GRID:-}"
ETA_GRID="${ETA_GRID:-}"

# =============================
# COMPUTE CHUNK BOUNDS
# =============================
# Count grid sizes to determine total triples.
if [[ -n "$SIGMA0_GRID" ]]; then
    N_S0=$(echo $SIGMA0_GRID | wc -w)
else
    if [[ "$FINE_GRID" == true ]]; then
        N_S0=15
    else
        N_S0=8
    fi
fi

if [[ -n "$SIGMA_GRID" ]]; then
    N_SIG=$(echo $SIGMA_GRID | wc -w)
else
    if [[ "$FINE_GRID" == true ]]; then
        N_SIG=13
    else
        N_SIG=7
    fi
fi

if [[ -n "$ETA_GRID" ]]; then
    N_ETA=$(echo $ETA_GRID | wc -w)
else
    if [[ "$FINE_GRID" == true ]]; then
        N_ETA=13
    else
        N_ETA=7
    fi
fi

TOTAL_TRIPLES=$((N_S0 * N_SIG * N_ETA))
CHUNK_SIZE=$(( (TOTAL_TRIPLES + N_JOBS - 1) / N_JOBS ))  # ceiling division
START=$(( SLURM_ARRAY_TASK_ID * CHUNK_SIZE ))
END=$(( START + CHUNK_SIZE ))
if [[ $END -gt $TOTAL_TRIPLES ]]; then
    END=$TOTAL_TRIPLES
fi

echo "======================================="
echo "SLURM_ARRAY_TASK_ID = $SLURM_ARRAY_TASK_ID"
echo "N_MC               = $N_MC"
echo "ISIS               = $ISIS"
echo "FINE_GRID          = $FINE_GRID"
echo "METRIC             = $METRIC"
echo "SEED               = $SEED"
echo "SAVE_DIR           = $SAVE_DIR"
echo "N_JOBS             = $N_JOBS"
echo "TOTAL_TRIPLES      = $TOTAL_TRIPLES"
echo "CHUNK_SIZE         = $CHUNK_SIZE"
echo "FLAT INDEX RANGE   = $START .. $((END - 1))"
[[ -n "$SIGMA0_GRID" ]] && echo "SIGMA0_GRID        = $SIGMA0_GRID"
[[ -n "$SIGMA_GRID" ]]  && echo "SIGMA_GRID         = $SIGMA_GRID"
[[ -n "$ETA_GRID" ]]    && echo "ETA_GRID           = $ETA_GRID"
echo "======================================="

# =============================
# BUILD GRID ARGS
# =============================
GRID_ARGS=()
if [[ -n "$SIGMA0_GRID" || -n "$SIGMA_GRID" || -n "$ETA_GRID" ]]; then
    [[ -n "$SIGMA0_GRID" ]] && GRID_ARGS+=(--sigma0-grid $SIGMA0_GRID)
    [[ -n "$SIGMA_GRID" ]]  && GRID_ARGS+=(--sigma-grid $SIGMA_GRID)
    [[ -n "$ETA_GRID" ]]    && GRID_ARGS+=(--eta-grid $ETA_GRID)
else
    [[ "$FINE_GRID" == true ]] && GRID_ARGS=(--fine)
fi

# =============================
# EXECUTION — loop over chunk of flat indices
# =============================
for (( FLAT_IDX=START; FLAT_IDX<END; FLAT_IDX++ )); do
    echo "--- Running flat index $FLAT_IDX / $((END - 1)) ---"
    python src/model/run_2d_grid_search_vectorized.py \
        --job-index "$FLAT_IDX" \
        --parallel-mode flat \
        "${GRID_ARGS[@]}" \
        --n-mc "$N_MC" \
        --isis $ISIS \
        --seed "$SEED" \
        --metric "$METRIC" \
        --n-sequences "$N_SEQUENCES" \
        --seq-length "$SEQ_LENGTH" \
        --min-pairs-per-isi "$MIN_PAIRS" \
        --save-dir "$SAVE_DIR" \
        --resume
done

echo "Done: chunk job $SLURM_ARRAY_TASK_ID (flat indices $START..$((END - 1)))"
