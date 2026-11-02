#!/bin/bash
#
#SBATCH --job-name=vjepa2_bib2_2gpu
#SBATCH --account=mrd344
#SBATCH --output=logs/vjepa2_bib2_2gpu_%j.out
#SBATCH --error=logs/vjepa2_bib2_2gpu_%j.err
#SBATCH --comment="preemption=yes;requeue=true"
#SBATCH --partition="c24m170-a100-2"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sy3494@nyu.edu

set -euo pipefail

mkdir -p logs

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

source /scratch/sy3494/BIB-VT_3-Copy1/.venv/bin/activate

# Make stdout/stderr reach the Slurm .out/.err files immediately.
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=12

nvidia-smi -L
nvidia-smi

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

SCRIPT="/scratch/sy3494/finetune_vjepa2_bib2_batched_2gpu.py"
DATA_ROOT="/scratch/sy3494/data/bib2_train_08_31_26"
OUTPUT_DIR="/scratch/sy3494/saved_models/vjepa2_bib2_2gpu_seed47_test"

# ------------------------------------------------------------
# Two-GPU benchmark
#
# --batch-size is PER GPU. With 2 GPUs, batch-size 8 gives a
# global effective batch of 16 examples per synchronized update.
#
# This first job intentionally uses only 200 train / 40 val tuples.
# Once the benchmark looks good, remove the two --max-* lines for
# a full-dataset epoch.
# ------------------------------------------------------------

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    "$SCRIPT" \
    --train-json-dir "$DATA_ROOT" \
    --val-json-dir "$DATA_ROOT" \
    --train-pattern "index_dict_train_*.json" \
    --val-pattern "index_dict_val_*.json" \
    --project-root /scratch/sy3494 \
    --mode mixed \
    --output-dir "$OUTPUT_DIR" \
    --epochs 6 \
    --batch-size 8 \
    --fam-microbatch-size 2 \
    --unfreeze-last-n 4 \
    --predictor-lr 1e-4 \
    --encoder-lr 1e-5 \
    --weight-decay 0.04 \
    --warmup-fraction 0.05 \
    --ema 0.99925 \
    --max-grad-norm 1.0 \
    --clip-frames 16 \
    --clip-fps 4 \
    --mask-ratio 0.60 \
    --n-fam-trials 8 \
    --frames-per-fam 16 \
    --test-fps 4 \
    --fam-end-trim-seconds 2 \
    --test-end-trim-seconds 2 \
    --mixed-masked-weight 0.5 \
    --seed 47 \
    --print-every 10 \
    --save-every-epoch 1 \

echo "Finished: $(date)"
