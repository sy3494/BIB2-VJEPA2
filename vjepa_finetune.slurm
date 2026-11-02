#!/bin/bash
#
#SBATCH --job-name=vjepa2_bib2_ft
#SBATCH --account=mrd344
#SBATCH --output=logs/vjepa2_bib2_ft_%j.out
#SBATCH --error=logs/vjepa2_bib2_ft_%j.err
#SBATCH --comment="preemption=yes;requeue=true;gpu_mps=yes"
#SBATCH --partition="c12m85-a100-1"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sy3494@nyu.edu

echo "Job: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------

source /scratch/sy3494/BIB-VT_3-Copy1/.venv/bin/activate

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

SCRIPT="/scratch/sy3494/finetune_vjepa2_bib2.py"

DATA_ROOT="/scratch/sy3494/data/bib2_train_08_31_26"

OUTPUT_DIR="/scratch/sy3494/saved_models/vjepa2_bib2_mixed_seed42"

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

python /scratch/sy3494/finetune_vjepa2_bib2_both_objectives.py \
    --train-json-dir /scratch/sy3494/data/bib2_train_08_31_26 \
    --val-json-dir /scratch/sy3494/data/bib2_train_08_31_26 \
    --train-pattern "index_dict_train_*.json" \
    --val-pattern "index_dict_val_*.json" \
    --project-root /scratch/sy3494 \
    --mode mixed \
    --output-dir /scratch/sy3494/saved_models/vjepa2_bib2_both_seed42_2 \
    --epochs 6 \
    --unfreeze-last-n 0 \
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
    --seed 42 \
    --print-every 100 \
    --save-every-epoch 1 \
    --max-train-examples 100 \
    --max-val-examples 20 \
    --print-every 1

echo "Finished: $(date)"


