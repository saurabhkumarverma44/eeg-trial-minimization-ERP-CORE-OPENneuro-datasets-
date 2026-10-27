
"""
stage3_labeling.py -- binary label assignment for Stage 3, reusing
CONDITION_SPEC from dl_prep/explicit_epoch_loader.py as the single
source of truth. No substring matching (that bug is what we're fixing).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dl_prep.explicit_epoch_loader import CONDITION_SPEC


def get_binary_label(dataset: str, task: str, file_id: str, raw_label: str, raw_code: int):
    """
    Returns 0, 1, or None (excluded) for a given epoch, using EXACT
    matching against CONDITION_SPEC -- never substring "in" checks.
    """
    spec_key = f"{dataset}_{task}"
    if spec_key not in CONDITION_SPEC:
        return None
    spec = CONDITION_SPEC[spec_key]

    file_stem = file_id.replace("-epo", "")
    if file_stem in spec.get("exclude", []):
        return None

    if "single_file_codes" in spec:  # N170: code-range based
        lo0, hi0 = spec["class0_range"]
        lo1, hi1 = spec["class1_range"]
        if lo0 <= raw_code <= hi0:
            return 0
        elif lo1 <= raw_code <= hi1:
            return 1
        return None

    if spec.get("digit_match_rule"):  # DS006018_visualoddball
        lo, hi = spec["valid_range"]
        if not (lo <= raw_code <= hi and raw_code < 100):
            return None
        tens, units = raw_code // 10, raw_code % 10
        return 0 if tens == units else (1 if tens != units else None)

    # Standard case: EXACT prefix match against class0/class1 names
    prefix = raw_label.split("/")[0]
    if prefix == spec.get("class0"):
        return 0
    elif prefix == spec.get("class1"):
        return 1
    return None
