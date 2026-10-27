import glob
import json
import os

print(f"{'Task Paradigm':<30} | {'Accuracy':<10} | {'F1-Score':<10} | {'AUC':<10}")
print("-" * 70)

for fpath in sorted(glob.glob("results_dl/figures/*/test_metrics.json")):
    folder_name = os.path.basename(os.path.dirname(fpath))
    with open(fpath) as f:
        m = json.load(f)
    
    acc_val = m.get("Test_Accuracy", 0)
    f1_val = m.get("F1_Score", 0)
    auc_val = m.get("AUC", "N/A")

    acc_str = f"{acc_val*100:.2f}%" if isinstance(acc_val, (int, float)) else str(acc_val)
    f1_str = f"{f1_val*100:.2f}%" if isinstance(f1_val, (int, float)) else str(f1_val)
    auc_str = f"{auc_val:.4f}" if isinstance(auc_val, (int, float)) else str(auc_val)
    
    print(f"{folder_name:<30} | {acc_str:<10} | {f1_str:<10} | {auc_str:<10}")
