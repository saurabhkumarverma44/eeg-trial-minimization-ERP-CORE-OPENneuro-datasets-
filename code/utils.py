"""Shared helpers: config loading, paths, and reproducible seeding."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or CONFIG_PATH) as fh:
        return yaml.safe_load(fh)


def resolve(rel_key: str, cfg: dict[str, Any] | None = None) -> Path:
    cfg = cfg or load_config()
    return PROJECT_ROOT / cfg["paths"][rel_key]


def set_seed(cfg: dict[str, Any] | None = None) -> int:
    cfg = cfg or load_config()
    seed = int(cfg["project"]["random_seed"])
    random.seed(seed)
    np.random.seed(seed)
    return seed