# preprocessing_batch/event_registry.py

# ==================================================
# ERP CORE
# ==================================================

ERP_CORE = {
    "P3": {
        "target": [
            11, 22, 33, 44, 55,
        ],
        "standard": [
            12, 13, 14, 15,
            21, 23, 24, 25,
            31, 32, 34, 35,
            41, 42, 43, 45,
            51, 52, 53, 54,
            201, 202,
        ],
    },
    "MMN": {
        "deviant": [70],
        "standard": [80],
    },
    "ERN": {
        "correct": [111, 121],
        "error": [212, 221, 222],
    },
    "N400": {
        "related": [111, 112, 121, 122],
        "unrelated": [211, 212, 221, 222],
    },
    "N2pc": {
        "left": [111, 112, 121, 122],
        "right": [211, 212, 221, 222],
    },
    "LRP": {
        "left": [11, 12, 21, 22],
        "right": [111, 121, 212, 221, 222],
    },
    # N170 has a large set of codes; for now, epoch on any event.
    "N170": {},  # use "all events" in preprocessing if needed
}

# ==================================================
# DS006018
# ==================================================

DS006018 = {
    "auditoryoddball": {
        "deviant": [70],
        "standard": [80],
    },
    "visualoddball": {
        "deviant": [70],
        "standard": [80],
    },
    # For now, leave flanker and visualsearch as "all events"
    "flanker": {},
    "visualsearch": {},
}

# ==================================================
# DS002680
# ==================================================

DS002680 = {
    "gonogo": {
        "target": [2],
        "distractor": [1],
        "correct": [3],
        "incorrect": [4],
    },
}