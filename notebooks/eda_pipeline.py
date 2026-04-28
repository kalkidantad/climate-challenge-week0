"""Shared profiling, cleaning, and plotting helpers for per-country EDA notebooks."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def repo_root_from_notebooks() -> Path:
    """Resolve project root whether the kernel cwd is repo root or notebooks/."""
    here = Path.cwd().resolve()
    cand = here.parent if here.name == "notebooks" else here
    if (cand / "data").is_dir():
        return cand
    if (here / "data").is_dir():
        return here
    return cand


def load_raw(country_slug: str, root: Path | None = None) -> pd.DataFrame:
    root = root or repo_root_from_notebooks()
    path = root / "data" / f"{country_slug}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Place the NASA POWER extract at {path} (see challenge data link). "
            "Do not commit CSVs to git."
        )
    # NASA POWER text exports may ship with a long parameter header (skip N lines).
    attempts: list[int | None] = [None]
    attempts.extend(list(range(18, 38)))
    last_err: Exception | None = None
    for skip in attempts:
        try:
            df = pd.read_csv(path, skiprows=skip) if skip is not None else pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
        if {"YEAR", "DOY"}.issubset(df.columns):
            return df
    raise ValueError(
        f"Could not find YEAR/DOY columns in {path}. "
        f"Try adjusting skiprows for your export. Last error: {last_err!r}"
    )


def initial_transform(df: pd.DataFrame, country_name: str) -> pd.DataFrame:
    out = df.copy()
    out["Country"] = country_name
    out["date"] = pd.to_datetime(
        out["YEAR"].astype(int) * 1000 + out["DOY"].astype(int), format="%Y%j"
    )
    out["Month"] = out["date"].dt.month
    return out


def replace_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    return df.replace(-999, np.nan)


def zscore_outlier_mask(
    df: pd.DataFrame, cols: list[str]
) -> pd.Series:
    mask = pd.Series(False, index=df.index)
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].astype(float)
        valid = s.dropna()
        if len(valid) < 2 or valid.nunique() < 2:
            continue
        z = np.abs(stats.zscore(s, nan_policy="omit"))
        m = z > 3
        m = m.reindex(df.index, fill_value=False)
        mask = mask | m.fillna(False)
    return mask


def handle_missing(
    df: pd.DataFrame, weather_cols: list[str], date_col: str = "date"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Drop rows where >30% of weather vars are NA; forward-fill remaining NAs in time order.
    Returns (cleaned_df, dropped_rows_df).
    """
    w = [c for c in weather_cols if c in df.columns]
    subset = df[w]
    frac = subset.isna().sum(axis=1) / max(len(w), 1)
    drop_idx = frac > 0.30
    dropped = df.loc[drop_idx]
    cleaned = df.loc[~drop_idx].copy()
    if date_col in cleaned.columns:
        cleaned = cleaned.sort_values(date_col)
    for c in w:
        cleaned[c] = cleaned[c].ffill()
    return cleaned, dropped


def monthly_means_plot(df: pd.DataFrame, country: str, ax_title: str = "") -> plt.Figure:
    monthly = (
        df.groupby(["YEAR", "Month"], as_index=False)["T2M"]
        .mean()
        .assign(
            period=lambda d: pd.to_datetime(
                d["YEAR"].astype(str) + "-" + d["Month"].astype(str).str.zfill(2) + "-01"
            )
        )
        .sort_values("period")
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(monthly["period"], monthly["T2M"], color="steelblue", linewidth=1.5)
    warm = monthly.loc[monthly["T2M"].idxmax()]
    cool = monthly.loc[monthly["T2M"].idxmin()]
    ax.scatter(warm["period"], warm["T2M"], color="red", s=80, zorder=5)
    ax.scatter(cool["period"], cool["T2M"], color="blue", s=80, zorder=5)
    ax.annotate(
        f"Warmest: {warm['T2M']:.1f}°C",
        (warm["period"], warm["T2M"]),
        textcoords="offset points",
        xytext=(10, 5),
        fontsize=9,
    )
    ax.annotate(
        f"Coolest: {cool['T2M']:.1f}°C",
        (cool["period"], cool["T2M"]),
        textcoords="offset points",
        xytext=(10, -15),
        fontsize=9,
    )
    ax.set_title(ax_title or f"Monthly average T2M — {country}")
    ax.set_ylabel("°C")
    ax.set_xlabel("Month (year-month)")
    fig.tight_layout()
    return fig


def monthly_precip_bars(df: pd.DataFrame, country: str) -> plt.Figure:
    monthly_p = (
        df.groupby(["YEAR", "Month"], as_index=False)["PRECTOTCORR"]
        .sum()
        .assign(
            period=lambda d: pd.to_datetime(
                d["YEAR"].astype(str) + "-" + d["Month"].astype(str).str.zfill(2) + "-01"
            )
        )
        .sort_values("period")
    )
    peak = monthly_p.loc[monthly_p["PRECTOTCORR"].idxmax()]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(monthly_p["period"], monthly_p["PRECTOTCORR"], width=20, color="teal", alpha=0.8)
    ax.axvline(peak["period"], color="orange", linestyle="--", label="Peak month")
    ax.annotate(
        f"Peak: {peak['PRECTOTCORR']:.1f} mm",
        (peak["period"], peak["PRECTOTCORR"]),
        textcoords="offset points",
        xytext=(10, 5),
        fontsize=9,
    )
    ax.set_title(f"Monthly total PRECTOTCORR — {country}")
    ax.set_ylabel("mm (sum of daily)")
    fig.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    num = df.select_dtypes(include=[np.number]).drop(columns=["YEAR", "DOY"], errors="ignore")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(num.corr(), annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Correlation matrix (numeric columns)")
    fig.tight_layout()
    return fig


def top_correlations(df: pd.DataFrame, n: int = 3) -> list[tuple[str, str, float]]:
    num = df.select_dtypes(include=[np.number]).drop(columns=["YEAR", "DOY"], errors="ignore")
    corr = num.corr()
    pairs: list[tuple[str, str, float]] = []
    cols = corr.columns.tolist()
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            v = corr.loc[a, b]
            if not np.isnan(v):
                pairs.append((a, b, float(v)))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    return pairs[:n]
