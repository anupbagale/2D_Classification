"""Reproducibility helpers."""

from __future__ import annotations

import json
import math
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _clean(obj):
    """Make numpy types and NaN JSON-safe (NaN -> null)."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, Path):
        return str(obj)
    return obj


def save_json(obj, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_clean(obj), f, indent=2)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def run_info(config: dict | None = None) -> dict:
    info = {
        "utc_time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "numpy": np.__version__,
    }
    try:
        import tensorflow as tf
        import keras

        info["tensorflow"] = tf.__version__
        info["keras"] = keras.__version__
        info["gpus"] = [d.name for d in tf.config.list_physical_devices("GPU")]
    except ImportError:
        pass
    try:
        import sklearn

        info["scikit_learn"] = sklearn.__version__
    except ImportError:
        pass
    if config:
        info["config"] = config
    return info
