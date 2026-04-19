#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=train_improved_%j.out
#SBATCH --job-name=bs_improved

# Activate environment
source ~/venv/bin/activate

cd ~/bs_detector

echo "=== Training with weighted CE ==="
python src/modeling/02_finetune_improved.py --loss weighted_ce

echo ""
echo "=== Evaluating weighted CE model ==="
python src/evaluation/02_evaluate_improved.py --run improved_weighted_ce_roberta-base_seed42

echo ""
echo "=== Done ==="
