import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/lp-dev/23BRS1328/erp_trial_min")
ML_DIR = PROJECT_ROOT / "stage3_results" / "ml"
IDX_DIR = PROJECT_ROOT / "stage3_results" / "subsample_indices"

STEPS = ["5", "10", "20", "30", "50", "75", "100", "150", "200", "full"]


class ParadigmData:
    """
    Loads X.npy / y.npy / metadata.csv / feature_block_boundaries.json for one
    paradigm folder under stage3_results/ml/, and indexes the matching
    subsample_indices JSON records for trial-count-restricted training.
    """

    def __init__(self, dataset_task: str):
        self.dataset_task = dataset_task
        d = ML_DIR / dataset_task
        self.X = np.load(d / "X.npy")
        self.y = np.load(d / "y.npy")
        self.metadata = pd.read_csv(d / "metadata.csv")
        with open(d / "feature_block_boundaries.json") as f:
            self.feature_blocks = json.load(f)
        self._index_subsample_records()

    def _index_subsample_records(self):
        dataset, task = self.dataset_task.split("__", 1)
        self.subject_step_indices = {}  # (subject, file_id) -> steps dict
        for jf in IDX_DIR.rglob("*.json"):
            rec = json.loads(jf.read_text())
            if rec.get("dataset") == dataset and rec.get("task") == task:
                key = (rec["subject"], rec["file_id"])
                self.subject_step_indices[key] = rec["steps"]

    def subjects(self):
        return sorted(self.metadata["subject"].unique())

    def all_row_indices_for_subject(self, subject: str) -> np.ndarray:
        return self.metadata.index[self.metadata["subject"] == subject].values

    def row_indices_for_subject_step(self, subject: str, step: str) -> np.ndarray:
        """
        ASSUMPTION (verify before trusting results): subsample_indices values
        are LOCAL indices keyed by epoch_index_in_subject within a given
        (subject, file_id), and map to global metadata rows via that column.
        """
        sub_meta = self.metadata[self.metadata["subject"] == subject]
        rows = []
        for (subj, file_id), steps in self.subject_step_indices.items():
            if subj != subject or step not in steps:
                continue
            local_idx = steps[step]
            file_meta = sub_meta[sub_meta["file_id"] == file_id]
            lookup = file_meta.set_index("epoch_index_in_subject")
            for i in local_idx:
                if i in lookup.index:
                    rows.append(lookup.loc[i].name)  # DataFrame index == row position
        return np.array(sorted(set(rows)), dtype=int)


def get_paradigms():
    return sorted([d.name for d in ML_DIR.iterdir() if d.is_dir()])