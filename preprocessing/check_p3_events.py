import pandas as pd

events = pd.read_csv(
    "data/raw/erp_core/sub-021/ses-P3/eeg/sub-021_ses-P3_task-P3_events.tsv",
    sep="\t"
)

critical = {11, 22, 33, 44, 55}

events["condition"] = events["value"].apply(
    lambda x: "critical" if x in critical else "standard"
)

print("\nCondition Counts")
print(events["condition"].value_counts())

print("\nCritical Codes")
print(sorted(events[events["condition"]=="critical"]["value"].unique()))

print("\nStandard Codes")
print(sorted(events[events["condition"]=="standard"]["value"].unique()))