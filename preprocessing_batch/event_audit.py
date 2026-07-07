from pathlib import Path
import pandas as pd

rows = []

# --------------------------------------------------
# ERP CORE
# --------------------------------------------------

for f in Path("data/raw/erp_core").rglob("*events.tsv"):

    try:
        df = pd.read_csv(f, sep="\t")

        if "value" in df.columns:
            events = sorted(df["value"].dropna().unique())

            rows.append({
                "dataset": "ERP_CORE",
                "file": str(f),
                "event_count": len(events),
                "events": ",".join(map(str, events[:50]))
            })

    except Exception as e:
        print("ERROR:", f)
        print(e)

# --------------------------------------------------
# DS006018
# --------------------------------------------------

for f in Path("data/raw/ds006018").rglob("*events.tsv"):

    try:
        df = pd.read_csv(f, sep="\t")

        cols = ",".join(df.columns)

        rows.append({
            "dataset": "DS006018",
            "file": str(f),
            "event_count": len(df),
            "events": cols
        })

    except Exception as e:
        print("ERROR:", f)
        print(e)

# --------------------------------------------------
# DS002680
# --------------------------------------------------

for f in Path("data/raw/ds002680").rglob("*events.tsv"):

    try:
        df = pd.read_csv(f, sep="\t")

        cols = ",".join(df.columns)

        rows.append({
            "dataset": "DS002680",
            "file": str(f),
            "event_count": len(df),
            "events": cols
        })

    except Exception as e:
        print("ERROR:", f)
        print(e)

outdir = Path("results_batch/audit")
outdir.mkdir(parents=True, exist_ok=True)

pd.DataFrame(rows).to_csv(
    outdir / "event_audit.csv",
    index=False
)

print("\nSaved:")
print(outdir / "event_audit.csv")
