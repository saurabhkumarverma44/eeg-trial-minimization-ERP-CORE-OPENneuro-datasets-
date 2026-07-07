from pathlib import Path

DATASETS = {
    "ERP_CORE": "data/raw/erp_core",
    "DS002680": "data/raw/ds002680",
    "DS006018": "data/raw/ds006018"
}

for dataset_name, root in DATASETS.items():

    root = Path(root)

    if not root.exists():
        print(f"{dataset_name}: NOT FOUND")
        continue

    subjects = sorted(
        [x.name for x in root.glob("sub-*") if x.is_dir()]
    )

    print("\n" + "="*60)
    print(dataset_name)
    print("="*60)

    print("Subjects:", len(subjects))

    for s in subjects[:10]:
        print(s)

    if len(subjects) > 10:
        print("...")
