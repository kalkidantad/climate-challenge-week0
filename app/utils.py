"""Load cleaned NASA POWER CSVs and build chart-ready frames (local `data/` only)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

COUNTRY_SLUGS = {
    "Ethiopia": "ethiopia",
    "Kenya": "kenya",
    "Sudan": "sudan",
    "Tanzania": "tanzania",
    "Nigeria": "nigeria",
}


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_cleaned_dataframe(countries: list[str]) -> pd.DataFrame:
    root = project_root()
    frames: list[pd.DataFrame] = []
    for name in countries:
        slug = COUNTRY_SLUGS[name]
        path = root / "data" / f"{slug}_clean.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run Task 2 notebooks to export `{slug}_clean.csv`."
            )
        d = pd.read_csv(path)
        d["Country"] = name
        if "date" not in d.columns:
            d["date"] = pd.to_datetime(
                d["YEAR"].astype(int) * 1000 + d["DOY"].astype(int), format="%Y%j"
            )
        else:
            d["date"] = pd.to_datetime(d["date"])
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def monthly_mean_t2m(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df[["Country", "date", "T2M"]].dropna(subset=["T2M"])
    return (
        tmp.groupby(["Country", pd.Grouper(key="date", freq="ME")])["T2M"]
        .mean()
        .reset_index()
    )


def filter_year_range(df: pd.DataFrame, y0: int, y1: int) -> pd.DataFrame:
    if df.empty:
        return df
    m = (df["YEAR"] >= y0) & (df["YEAR"] <= y1)
    return df.loc[m].reset_index(drop=True)


def pivot_variable_monthly_mean(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Monthly mean for an arbitrary numeric column (dashboard variable selector)."""
    tmp = df[["Country", "date", col]].dropna(subset=[col])
    agg = tmp.groupby(["Country", pd.Grouper(key="date", freq="ME")])[col].mean().reset_index()
    return agg
