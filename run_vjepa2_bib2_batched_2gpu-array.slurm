#!/bin/bash
#
#SBATCH --job-name=vjepa2_bib2_ft
#SBATCH --account=mrd344
#SBATCH --output=logs/vjepa2_bib2_ft_%A_%a.out
#SBATCH --error=logs/vjepa2_bib2_ft_%A_%a.err
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --partition="c24m170-a100-2"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --array=0-4
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sy3494@nyu.edu
#SBATCH --exclude=b-20-11

# ------------------------------------------------------------
# Seed for this array task
# ------------------------------------------------------------

SEEDS=(1234 1235 1236 1237 1238)
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi -L
nvidia-smi

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

SCRIPT="/scratch/sy3494/finetune_vjepa2_bib2_batched_2gpu.py"
DATA_ROOT="/scratch/sy3494/data/bib2_train_08_31_26"

OUTPUT_DIR="/scratch/sy3494/saved_models/vjepa2_bib2_2gpu_seed${SEED}"

mkdir -p "$OUTPUT_DIR"

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

torchrun \
    --standalone \
    --nproc_per_node=2 \
    "$SCRIPT" \
    --train-json-dir "$DATA_ROOT" \
    --val-json-dir "$DATA_ROOT" \
    --train-pattern "index_dict_train_*.json" \
    --val-pattern "index_dict_val_*.json" \
    --project-root /scratch/sy3494 \
    --mode mixed \
    --output-dir "$OUTPUT_DIR" \
    --epochs 4 \
    --batch-size 8 \
    --fam-microbatch-size 2 \
    --unfreeze-last-n 4 \
    --predictor-lr 1e-4 \
    --encoder-lr 1e-5 \
    --weight-decay 0.04 \
    --ema 0.99925 \
    --clip-frames 16 \
    --clip-fps 4 \
    --mask-ratio 0.60 \
    --n-fam-trials 8 \
    --frames-per-fam 16 \
    --test-fps 4 \
    --fam-end-trim-seconds 2 \
    --test-end-trim-seconds 2 \
    --mixed-masked-weight 0.5 \
    --seed "$SEED" \
    --print-every 10 \
    --save-every-epoch 1

echo "============================================"
echo "Seed $SEED finished: $(date)"
echo "============================================"