#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=train_focal_%j.out
#SBATCH --job-name=bs_focal

# Activate environment
source ~/venv/bin/activate

cd ~/bs_detector

echo "=== Training with focal loss ==="
python src/modeling/02_finetune_improved.py --loss focal

echo ""
echo "=== Evaluating focal loss model ==="
python src/evaluation/02_evaluate_improved.py --run improved_focal_roberta-base_seed42

echo ""
echo "=== Done ==="
