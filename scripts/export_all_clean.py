#!/usr/bin/env python3
"""
Batch-export data/<country>_clean.csv for all five countries (same logic as Task 2 notebooks).

Run from repo root:
  python scripts/export_all_clean.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "notebooks"))

import numpy as np  # noqa: E402
from eda_pipeline import (  # noqa: E402
    handle_missing,
    initial_transform,
    load_raw,
    replace_sentinels,
)

PAIR = [
    ("ethiopia", "Ethiopia"),
    ("kenya", "Kenya"),
    ("sudan", "Sudan"),
    ("tanzania", "Tanzania"),
    ("nigeria", "Nigeria"),
]

WEATHER = [
    "T2M",
    "T2M_MAX",
    "T2M_MIN",
    "T2M_RANGE",
    "PRECTOTCORR",
    "RH2M",
    "WS2M",
    "WS2M_MAX",
    "PS",
    "QV2M",
]


def main() -> None:
    (ROOT / "data").mkdir(parents=True, exist_ok=True)
    for slug, name in PAIR:
        print(f"Processing {slug}.csv …", end=" ")
        df = load_raw(slug, ROOT)
        df = initial_transform(df, name)
        df = replace_sentinels(df)
        if df.duplicated().any():
            df = df.drop_duplicates().reset_index(drop=True)
        df_clean, dropped = handle_missing(df, WEATHER)
        out = ROOT / "data" / f"{slug}_clean.csv"
        df_clean.to_csv(out, index=False)
        print(f"→ {out.name} (rows={len(df_clean)}, dropped sparse={len(dropped)})")


if __name__ == "__main__":
    main()
