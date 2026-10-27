#!/bin/bash
set -e
echo "================================================================"
echo "STARTING FULL PIPELINE: PHASE 3 & PHASE 4 FOR ALL 3 DATASETS"
echo "================================================================"
# Step 1: Run Phase 3 Data Preparation for all 3 datasets
echo "[1/2] Running Phase 3 Data Preparation..."
python run_phase3_prep.py --all_datasets
# Step 2: Run Phase 4 1D CNN Training across all 3 datasets
echo "[2/2] Running Phase 4 1D CNN Baseline Training..."
python run_phase4_train_cnn.py --all_datasets --epochs 50 --batch_size 32
echo "================================================================"
echo "ALL 3 DATASETS PREPARED AND TRAINED SUCCESSFULLY!"
echo "Check results in /lp-dev/23BRS1328/erp_trial_min/results_dl"
echo "================================================================"
