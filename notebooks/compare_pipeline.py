"""Cross-country aggregation and plotting for compare_countries notebook."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


SLUGS = ["ethiopia", "kenya", "sudan", "tanzania", "nigeria"]
LABELS = {
    "ethiopia": "Ethiopia",
    "kenya": "Kenya",
    "sudan": "Sudan",
    "tanzania": "Tanzania",
    "nigeria": "Nigeria",
}


def load_all_clean(root: Path | None = None) -> pd.DataFrame:
    root = root or Path(__file__).resolve().parent.parent
    frames = []
    for slug in SLUGS:
        p = root / "data" / f"{slug}_clean.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing {p}. Run per-country EDA notebooks first.")
        d = pd.read_csv(p)
        d["Country"] = LABELS[slug]
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def add_date(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(
        out["YEAR"].astype(int) * 1000 + out["DOY"].astype(int), format="%Y%j"
    )
    out["Month"] = out["date"].dt.month
    return out


def monthly_t2m_overlay(df: pd.DataFrame) -> plt.Figure:
    d = add_date(df)
    monthly = (
        d.groupby(["Country", "YEAR", "Month"], as_index=False)["T2M"]
        .mean()
        .assign(
            period=lambda x: pd.to_datetime(
                x["YEAR"].astype(str) + "-" + x["Month"].astype(str).str.zfill(2) + "-01"
            )
        )
        .sort_values(["Country", "period"])
    )
    fig, ax = plt.subplots(figsize=(13, 5))
    for c in sorted(monthly["Country"].unique()):
        sub = monthly[monthly["Country"] == c]
        ax.plot(sub["period"], sub["T2M"], label=c, linewidth=1.4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))
    ax.set_title("Monthly average T2M — all countries (NASA POWER points)")
    ax.set_ylabel("°C")
    fig.tight_layout()
    return fig


def agg_table_t2m(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Country")["T2M"].agg(["mean", "median", "std"])
    return g.round(3)


def agg_table_precip(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("Country")["PRECTOTCORR"].agg(["mean", "median", "std"])
    return g.round(3)


def precip_boxplots(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    order = sorted(df["Country"].unique())
    sns.boxplot(data=df, x="Country", y="PRECTOTCORR", order=order, ax=ax)
    ax.set_title("Daily PRECTOTCORR by country")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    return fig


def extreme_heat_days_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """Count days per YEAR and Country where T2M_MAX > 35."""
    mask = df["T2M_MAX"] > 35
    sub = df.loc[mask]
    return (
        sub.groupby(["Country", "YEAR"], as_index=False)
        .size()
        .rename(columns={"size": "heat_days_35"})
    )


def max_consecutive_dry_streak_daily(sub_year: pd.DataFrame) -> int:
    s = sub_year.sort_values("DOY")["PRECTOTCORR"].values
    streak = mx = 0
    for p in s:
        if p < 1:
            streak += 1
            mx = max(mx, streak)
        else:
            streak = 0
    return mx


def max_dry_days_by_year_country(df: pd.DataFrame) -> pd.DataFrame:
    d = add_date(df)
    rows = []
    for (c, y), grp in d.groupby(["Country", "YEAR"]):
        rows.append(
            {
                "Country": c,
                "YEAR": y,
                "max_consecutive_dry_days": max_consecutive_dry_streak_daily(grp.sort_values("date")),
            }
        )
    return pd.DataFrame(rows)


def anova_or_kruskal_t2m(df: pd.DataFrame) -> tuple[str, float, float, object]:
    """Return test name, statistic, p-value, and scipy result object."""
    groups = [
        df.loc[df["Country"] == c, "T2M"].dropna().values
        for c in sorted(df["Country"].unique())
    ]
    norm_ok = len(groups) >= 2 and all(len(g) >= 20 for g in groups)
    if norm_ok:
        lev = stats.levene(*groups)
        if getattr(lev, "pvalue", 1.0) < 0.05:
            h = stats.kruskal(*groups)
            return ("Kruskal–Wallis (heterogeneous variance)", float(h.statistic), float(h.pvalue), h)
        f = stats.f_oneway(*groups)
        return ("One-way ANOVA", float(f.statistic), float(f.pvalue), f)
    h = stats.kruskal(*groups)
    return ("Kruskal–Wallis", float(h.statistic), float(h.pvalue), h)


def vulnerability_rank_summary(
    df: pd.DataFrame, heat_counts: pd.DataFrame, drought: pd.DataFrame
) -> pd.DataFrame:
    """Composite ranking: higher warmth, precip variability, heat days, dry streaks → higher stress."""
    tc = agg_table_t2m(df).rename(columns={"mean": "t2m_mean", "std": "t2m_std"})
    pc = agg_table_precip(df).rename(columns={"std": "precip_std"})
    countries = sorted(df["Country"].unique())
    heat_mean = (
        heat_counts.groupby("Country")["heat_days_35"].mean()
        if not heat_counts.empty
        else pd.Series(0.0, index=countries)
    )
    heat_mean = heat_mean.reindex(countries, fill_value=0.0).rename("heat_days_mean_ann")
    d_mean = drought.groupby("Country")["max_consecutive_dry_days"].mean().rename("dry_streak_mean")
    d_mean = d_mean.reindex(countries).fillna(0.0)
    out = tc.join(pc[["precip_std"]], how="outer").join(heat_mean, how="left").join(d_mean, how="left")
    for col in out.columns:
        out[col] = out[col].fillna(out[col].median())
    out["stress_score"] = (
        z(out["t2m_mean"])
        + z(out["precip_std"])
        + z(out["heat_days_mean_ann"])
        + z(out["dry_streak_mean"])
    )
    out["rank_most_vulnerable"] = out["stress_score"].rank(ascending=False, method="min")
    return out.round(4).sort_values("stress_score", ascending=False)


def z(s: pd.Series) -> pd.Series:
    x = s.astype(float)
    if len(x) < 2 or x.std(ddof=0) == 0 or pd.isna(x.std(ddof=0)):
        return pd.Series(0.0, index=s.index)
    v = stats.zscore(x, nan_policy="omit")
    return pd.Series(v, index=s.index)


def bar_by_country(metric_df: pd.DataFrame, value_col: str, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = metric_df.groupby(["Country", "YEAR"])[value_col].first().reset_index()
    countries = sorted(pivot["Country"].unique())
    years = sorted(pivot["YEAR"].unique())
    x = np.arange(len(years))
    w = 0.75 / len(countries)
    for i, c in enumerate(countries):
        sub = pivot[pivot["Country"] == c].set_index("YEAR").reindex(years)[value_col].fillna(0)
        ax.bar(x + i * w, sub.values, width=w, label=c)
    ax.set_xticks(x + w * (len(countries) / 2 - 0.5))
    ax.set_xticklabels(years.astype(int))
    ax.legend(title="Country")
    ax.set_title(title)
    fig.tight_layout()
    return fig
