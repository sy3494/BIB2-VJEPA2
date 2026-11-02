#!/bin/bash
#SBATCH --job-name=vjepa2_bib2_eval
#SBATCH --account=mrd344
#SBATCH --output=logs/vjepa2_bib2_eval_%A_%a.out
#SBATCH --error=logs/vjepa2_bib2_eval_%A_%a.err
#SBATCH --partition="c12m85-a100-1"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=3
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sy3494@nyu.edu

set -euo pipefail

# ------------------------------------------------------------
# Seed for this array task
# ------------------------------------------------------------

SEEDS=(42 43 44 45 46)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "============================================"
echo "Job:        $SLURM_JOB_ID"
echo "Array job:  $SLURM_ARRAY_JOB_ID"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Seed:       $SEED"
echo "Node:       $(hostname)"
echo "Started:    $(date)"
echo "============================================"

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

source /scratch/sy3494/BIB-VT_3-Copy1/.venv/bin/activate

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

SCRIPT="/scratch/sy3494/vjepa2_bib2_eval_final_5seeds_fallback_epoch2.py"

JSON_DIR="/scratch/sy3494/data/bib2_eval_infant"
MODEL_ROOT="/scratch/sy3494/saved_models"

RESULT_DIR="/scratch/sy3494/results/vjepa2_bib2_eval"
mkdir -p "$RESULT_DIR"

OUTPUT="${RESULT_DIR}/vjepa2_bib2_seed${SEED}.csv"

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

python -u "$SCRIPT" \
    --json-dir "$JSON_DIR" \
    --pattern "index_dict_eval_*.json" \
    --project-root /scratch/sy3494 \
    --model-root "$MODEL_ROOT" \
    --seeds "$SEED" \
    --n-fam-trials 8 \
    --frames-per-fam 90 \
    --test-fps 25 \
    --fam-end-trim-seconds 2 \
    --test-end-trim-seconds 2 \
    --test-stride 1 \
    --device cuda \
    --output "$OUTPUT"

echo "============================================"
echo "Seed $SEED complete"
echo "Output: $OUTPUT"
echo "Finished: $(date)"
echo "============================================"