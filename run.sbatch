#!/bin/bash
#SBATCH --job-name=introspection
# Set your cluster's partition with sbatch --partition=...
#SBATCH --gres=gpu:a100:1
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.out
#
# Full pipeline (build vector -> calibrate -> persona matrix) on the default 7B model:
#   sbatch run.sbatch
# A bigger model needs more GPUs and a proportionally deeper layer, e.g. 40 of 80:
#   MODEL=Qwen/Qwen2.5-72B-Instruct LAYER=40 SCALE=60 sbatch --gres=gpu:a100:4 --mem=120G run.sbatch
# Env overrides: MODEL, LAYER, SCALES (sweep), SCALE (matrix), RESULTS (output dir), HF_CACHE.
set -euo pipefail

# Optional custom Hugging Face cache.
if [[ -n ${HF_CACHE:-} ]]; then export HF_HOME="$HF_CACHE"; fi
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
PY=${PY:-python}

MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
if [[ "$MODEL" == "Qwen/Qwen2.5-7B-Instruct" ]]; then
    LAYER=${LAYER:-14}
    SCALE=${SCALE:-20}
elif [[ "$MODEL" == "Qwen/Qwen2.5-72B-Instruct" ]]; then
    LAYER=${LAYER:-40}
    SCALE=${SCALE:-60}
else
    : "${LAYER:?Set LAYER explicitly for a new model (zero-indexed decoder block)}"
fi
RESULTS=${RESULTS:-results/$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's/-instruct$//')}

cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"
export PYTHONPATH=.

"$PY" scripts/build_vector.py --model "$MODEL" --layer "$LAYER" --results "$RESULTS"
"$PY" scripts/calibrate.py --results "$RESULTS" --scales ${SCALES:-0 20 40 60 80 100}
if [[ -n ${SCALE:-} ]]; then
    "$PY" scripts/persona_matrix.py --results "$RESULTS" --scale "$SCALE"
else
    echo "Calibration saved. Inspect calibration.json, then run persona_matrix.py with an explicit --scale."
fi
