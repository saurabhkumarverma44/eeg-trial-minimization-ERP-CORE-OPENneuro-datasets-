import pandas as pd
import os

csv_path = "results_dl/trial_minimization/summary_minimization.csv"

if not os.path.exists(csv_path):
    print(f"[Error] {csv_path} not found. Please verify Phase 5 output files.")
else:
    df = pd.read_csv(csv_path)
    
    print("\n" + "="*110)
    print(f"{'1D CNN TRIAL MINIMIZATION FRAMEWORK - OVERALL SUMMARY':^108}")
    print("="*110)
    
    header = f"{'Task Paradigm':<25} | {'Baseline N':<10} | {'Base Acc (%)':<12} | {'Min Stable N':<12} | {'Min Acc (%)':<11} | {'Trial Reduction (%)':<18}"
    print(header)
    print("-" * 110)
    
    for idx, row in df.iterrows():
        name = row['target_name']
        base_n = int(row['baseline_trials'])
        
        base_acc = row['baseline_accuracy'] * 100.0 if row['baseline_accuracy'] <= 1.0 else row['baseline_accuracy']
        min_n = int(row['min_stable_trials'])
        min_acc = row['accuracy_at_min_trials'] * 100.0 if row['accuracy_at_min_trials'] <= 1.0 else row['accuracy_at_min_trials']
        red_pct = row['trial_reduction_percentage']
        
        print(f"{name:<25} | {base_n:<10} | {base_acc:<12.2f} | {min_n:<12} | {min_acc:<11.2f} | {red_pct:<18.2f}%")
        
    print("-" * 110)
    
    # Calculate dataset-wide metrics
    avg_base_acc = df['baseline_accuracy'].apply(lambda x: x*100.0 if x <= 1.0 else x).mean()
    avg_min_acc = df['accuracy_at_min_trials'].apply(lambda x: x*100.0 if x <= 1.0 else x).mean()
    avg_red = df['trial_reduction_percentage'].mean()
    
    print(f"{'MEAN OVERALL':<25} | {'-':<10} | {avg_base_acc:<12.2f} | {'-':<12} | {avg_min_acc:<11.2f} | {avg_red:<18.2f}%")
    print("="*110 + "\n")
