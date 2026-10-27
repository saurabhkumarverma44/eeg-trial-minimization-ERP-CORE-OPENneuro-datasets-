from pathlib import Path
import pandas as pd

rows = []

# ERP CORE
for subj in Path("data/raw/erp_core").glob("sub-*"):

    sessions = sorted(
        [x.name for x in subj.glob("ses-*")]
    )

    rows.append({
        "dataset": "ERP_CORE",
        "subject": subj.name,
        "items": ",".join(sessions)
    })

# DS006018
for subj in Path("data/raw/ds006018").glob("sub-*"):

    tasks = set()

    for vhdr in subj.rglob("*.vhdr"):

        name = vhdr.name

        if "task-" in name:

            task = (
                name
                .split("task-")[1]
                .split("_")[0]
            )

            tasks.add(task)

    rows.append({
        "dataset": "DS006018",
        "subject": subj.name,
        "items": ",".join(sorted(tasks))
    })

# DS002680
for subj in Path("data/raw/ds002680").glob("sub-*"):

    sessions = sorted(
        [x.name for x in subj.glob("ses-*")]
    )

    rows.append({
        "dataset": "DS002680",
        "subject": subj.name,
        "items": ",".join(sessions)
    })

out = pd.DataFrame(rows)

Path(
    "results_batch/audit"
).mkdir(
    parents=True,
    exist_ok=True
)

out.to_csv(
    "results_batch/audit/task_audit.csv",
    index=False
)

print(
    "Saved: results_batch/audit/task_audit.csv"
)
