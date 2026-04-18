#!/bin/bash
#SBATCH --job-name=nlp-sentiment
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=cluster_output_%j.log

echo "======================================================="
echo " NLP Sentiment Pipeline - GPU Job"
echo " Node: $(hostname) | GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'checking...')"
echo "======================================================="

# Activate environment (change 'nlp' to your env name)
source $HOME/miniforge3/bin/activate /scratch/$USER/nlp 2>/dev/null \
    || source $HOME/miniforge3/bin/activate base

cd ~/nlp-project

echo ">>> Step 1: Feature Extraction"
python feature_extraction.py

echo ">>> Step 2: Model Training (LogReg + RoBERTa)"
python train_models.py

echo ">>> Step 3: Evaluation"
python evaluate_models.py

echo ">>> Step 4: Error Analysis"
python error_analysis.py

echo "======================================================="
echo " Pipeline complete!"
echo "======================================================="
