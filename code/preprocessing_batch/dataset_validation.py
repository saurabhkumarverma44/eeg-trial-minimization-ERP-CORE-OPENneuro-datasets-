from pathlib import Path
import pandas as pd

DATASETS = {
    "ERP_CORE": Path("data/raw/erp_core"),
    "DS006018": Path("data/raw/ds006018"),
    "DS002680": Path("data/raw/ds002680")
}

rows = []

for dataset_name, root in DATASETS.items():

    subjects = sorted(
        [x for x in root.glob("sub-*") if x.is_dir()]
    )

    rows.append({
        "dataset": dataset_name,
        "subjects": len(subjects)
    })

    print("\n" + "="*60)
    print(dataset_name)
    print("="*60)

    print("Subjects:", len(subjects))

    for s in subjects[:5]:
        print(s.name)

summary = pd.DataFrame(rows)

Path("results_batch/audit").mkdir(
    parents=True,
    exist_ok=True
)

summary.to_csv(
    "results_batch/audit/dataset_summary.csv",
    index=False
)

print("\nSaved:")
print("results_batch/audit/dataset_summary.csv")
