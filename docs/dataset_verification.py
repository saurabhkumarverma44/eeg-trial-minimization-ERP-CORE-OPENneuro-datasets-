from pathlib import Path

ROOT = Path("data/raw")

print("=" * 60)
print("PHASE 1 DATASET VERIFICATION")
print("=" * 60)

datasets = {
    "ERP_CORE": ROOT / "erp_core",
    "ds006018": ROOT / "ds006018",
    "ds002680": ROOT / "ds002680"
}

for name, path in datasets.items():

    print(f"\n{name}")
    print("-" * 40)

    if not path.exists():
        print("Dataset not found")
        continue

    subjects = sorted(
        [x.name for x in path.iterdir()
         if x.is_dir() and x.name.startswith("sub-")]
    )

    print("Subjects Found :", len(subjects))

    if len(subjects) > 0:
        print("First Subject  :", subjects[0])
        print("Last Subject   :", subjects[-1])

print("\nVerification Complete")