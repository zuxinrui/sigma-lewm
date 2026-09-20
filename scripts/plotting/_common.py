"""Shared plotting style for analysis scripts (CPU only)."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

STYLE = Path(__file__).resolve().parent


def palette() -> dict:
    return json.loads((STYLE / "palette.json").read_text())


def apply_style() -> dict:
    plt.style.use(str(STYLE / "paper.mplstyle"))
    return palette()
