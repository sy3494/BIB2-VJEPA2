#!/bin/bash
#
#SBATCH --job-name=BIB-VJEPA2_Eval
#SBATCH --account=mrd344
#SBATCH --comment="preemption=yes;requeue=true;gpu_mps=yes"
#SBATCH --output=/scratch/sy3494/BIB-VJEPA2_Eval_long_fix.out
#SBATCH --partition="c12m85-a100-1"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=5:00:00
#SBATCH --mem=40GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sy3494@nyu.edu

source /scratch/sy3494/BIB-VT_3-Copy1/.venv/bin/activate

python vjepa2_bib2_first4_bib_eval.py --json-dir "/scratch/sy3494/data/bib2_eval_infant" --project-root "/scratch/sy3494/data/bib2_eval_infant" --output vjepa2_bib2_results.csv


