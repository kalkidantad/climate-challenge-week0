#!/usr/bin/env python3
"""Generate Week 0 challenge Jupyter notebooks (EDA x5 + compare_countries). Run from repo root."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def write_ipynb(path: Path, cells: list[dict]) -> None:
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(nb, indent=1), encoding="utf-8")


def md(text: str) -> dict:
    src = text if text.endswith("\n") else text + "\n"
    return {"cell_type": "markdown", "metadata": {}, "source": [src]}


def code(src: str) -> dict:
    if not src.endswith("\n"):
        src += "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [src],
    }


def eda_notebook(slug: str, title: str) -> list[dict]:
    return [
        md(
            f"# {title} — NASA POWER EDA (Week 0 Task 2)\n\n"
            f"Place the official extract at **`data/{slug}.csv`** (not committed). "
            "See the challenge brief for download instructions."
        ),
        code(
            f"""import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "notebooks"))

from eda_pipeline import (
    correlation_heatmap,
    handle_missing,
    initial_transform,
    load_raw,
    monthly_means_plot,
    monthly_precip_bars,
    replace_sentinels,
    repo_root_from_notebooks,
    top_correlations,
    zscore_outlier_mask,
)

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_palette("deep")

COUNTRY_SLUG = "{slug}"
COUNTRY_NAME = "{title}"
"""
        ),
        md("## Data loading & date parsing"),
        code(
            f"""ROOT = repo_root_from_notebooks()
df = load_raw(COUNTRY_SLUG, ROOT)
print("shape", df.shape, "columns:", list(df.columns))
df = initial_transform(df, COUNTRY_NAME)
df.head()"""
        ),
        md(
            "## NASA sentinel `-999`, duplicates\n\n"
            "Replace `-999` with NaN **before** summary statistics ([NASA POWER missing values]"
            "(https://power.larc.nasa.gov/docs/tutorials/quick-data-analysis-python/)).\n\n"
            "Duplicate rows typically repeat identical values across **all** data columns."
        ),
        code(
            """df = replace_sentinels(df)

ndup = int(df.duplicated().sum())
print("duplicate rows (full-row duplicates):", ndup)

if ndup > 0:
    dup_mask = df.duplicated(keep=False)
    cols_all = df.columns.tolist()
    print("duplicate investigation: rows share identical values across:", cols_all[:12], "...")
    display(df.loc[dup_mask].sort_values(by=cols_all).head(15))
    df = df.drop_duplicates().reset_index(drop=True)
    print("after drop_duplicates:", df.shape)
else:
    print("No duplicate rows.")

desc = df.describe()
display(desc)

na = df.isna().sum().sort_values(ascending=False)
pct = (na / len(df) * 100).round(2)
missing_report = pd.DataFrame({"count": na, "pct": pct}).loc[lambda x: x["count"] > 0]
missing_report = missing_report.sort_values("count", ascending=False)
display(missing_report)
high = missing_report[missing_report["pct"] > 5]
print("columns with >5% missing:", list(high.index) if len(high) else "none")"""
        ),
        md(
            "### Brief interpretation (`describe`)\n\n"
            "- **T2M / T2M_MAX / T2M_MIN:** central tendency and spread for the single grid point.\n"
            "- **PRECTOTCORR:** many dry-day zeros — mean can be small while wet days drive totals.\n"
            "- **RH2M / QV2M / PS:** QC sanity (pressure should stay in a plausible surface range)."
        ),
        md(
            "## Outlier screening — |Z| > 3\n\n"
            "Applied to: **T2M**, **T2M_MAX**, **T2M_MIN**, **PRECTOTCORR**, **RH2M**, **WS2M**, **WS2M_MAX**."
        ),
        code(
            """zcols = ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "RH2M", "WS2M", "WS2M_MAX"]
mask = zscore_outlier_mask(df, zcols)
print("rows with |Z|>3 in any listed column:", int(mask.sum()))
if mask.any():
    display(df.loc[mask].head(10))"""
        ),
        md(
            "### Decision: retain, cap, or drop?\n\n"
            "**Retain** flagged rows for this EDA. Extreme temperatures or rainfall often reflect real "
            "weather (heat waves, cloudbursts) that matter for COP-style narratives. Capping would hide "
            "tails; dropping would remove rare events. We only treat **missing** data below."
        ),
        md(
            "## Handle missing values\n\n"
            "- Drop rows where **>30%** of the weather columns listed are missing.\n"
            "- **Forward-fill** remaining gaps in time order for weather variables."
        ),
        code(
            """weather_cols = [
    "T2M", "T2M_MAX", "T2M_MIN", "T2M_RANGE", "PRECTOTCORR",
    "RH2M", "WS2M", "WS2M_MAX", "PS", "QV2M",
]
df_clean, dropped_rows = handle_missing(df, weather_cols)
print("rows dropped (>30% missing in weather subset):", len(dropped_rows))
out_path = ROOT / "data" / f"{COUNTRY_SLUG}_clean.csv"
out_path.parent.mkdir(parents=True, exist_ok=True)
df_clean.to_csv(out_path, index=False)
print("exported:", out_path.resolve())"""
        ),
        md("## Time series — monthly T2M & monthly total precipitation"),
        code(
            """fig = monthly_means_plot(df_clean, COUNTRY_NAME)
plt.show()
fig2 = monthly_precip_bars(df_clean, COUNTRY_NAME)
plt.show()"""
        ),
        md(
            "### Trends / anomalies (commentary)\n\n"
            "Scan the T2M line for multi-year drift vs interannual variability. "
            "Compare wet seasons via monthly PRECTOTCORR totals. "
            "This is exploratory — attribution needs multi-model ensembles and methods beyond raw point data."
        ),
        md("## Correlations"),
        code(
            """fig = correlation_heatmap(df_clean)
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].scatter(df_clean["T2M"], df_clean["RH2M"], s=4, alpha=0.35)
ax[0].set_xlabel("T2M (°C)")
ax[0].set_ylabel("RH2M (%)")
ax[0].set_title("T2M vs RH2M")
ax[1].scatter(df_clean["T2M_RANGE"], df_clean["WS2M"], s=4, alpha=0.35)
ax[1].set_xlabel("T2M_RANGE (°C)")
ax[1].set_ylabel("WS2M (m/s)")
ax[1].set_title("Diurnal range vs wind")
plt.tight_layout()
plt.show()

for a, b, r in top_correlations(df_clean, 3):
    print(f"{a} ↔ {b}: r = {r:.3f}")"""
        ),
        md(
            "### Three strongest correlations\n\n"
            "Summarize the printed pairs: e.g. temperature variables co-move; "
            "humidity often drops as daily mean temperature rises in this sample."
        ),
        md("## Distributions — rainfall histogram & bubble chart"),
        code(
            """p = df_clean["PRECTOTCORR"].clip(lower=0)
skew = float(p.skew())

fig, ax = plt.subplots(figsize=(7, 4))
if skew > 2:
    ax.hist(np.log1p(p), bins=60, color="steelblue", edgecolor="white")
    ax.set_xlabel("log(1 + PRECTOTCORR)")
    ax.set_title(f"PRECTOTCORR — log1p scale (skew={skew:.2f})")
else:
    ax.hist(p, bins=60, color="steelblue", edgecolor="white")
    ax.set_xlabel("PRECTOTCORR (mm/day)")
plt.ylabel("count")
plt.tight_layout()
plt.show()

sample = df_clean.sample(min(2500, len(df_clean)), random_state=0)
sizes = np.clip(sample["PRECTOTCORR"] * 5, 1, 120)
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(
    sample["T2M"],
    sample["RH2M"],
    s=sizes,
    alpha=0.35,
    c=sample["Month"],
    cmap="viridis",
)
ax.set_xlabel("T2M (°C)")
ax.set_ylabel("RH2M (%)")
ax.set_title("Bubble size = PRECTOTCORR (mm/day)")
plt.colorbar(sc, ax=ax, label="Month")
plt.tight_layout()
plt.show()"""
        ),
        md(
            "## References used\n\n"
            "- NASA POWER data access: https://power.larc.nasa.gov/\n"
            "- WMO *State of the Climate in Africa*: https://wmo.int/publication-series/state-of-climate-africa-2024\n"
            "- World Bank climate risk profiles: https://climateknowledgeportal.worldbank.org/\n"
        ),
    ]


COMPARE_CELLS = [
    md(
        "# Cross-country comparison — Week 0 Task 3\n\n"
        "Reads `data/ethiopia_clean.csv` … `data/nigeria_clean.csv` produced in Task 2."
    ),
    code(
        """import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display

ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "notebooks"))

from compare_pipeline import (
    agg_table_precip,
    agg_table_t2m,
    anova_or_kruskal_t2m,
    bar_by_country,
    extreme_heat_days_by_year,
    load_all_clean,
    max_dry_days_by_year_country,
    monthly_t2m_overlay,
    precip_boxplots,
    vulnerability_rank_summary,
)

sns.set_theme(style="whitegrid")
df = load_all_clean(ROOT)
display(df["Country"].value_counts())
df.head()"""
    ),
    md("## Monthly average T2M — all countries + summary table"),
    code(
        """fig = monthly_t2m_overlay(df)
plt.tight_layout()
plt.show()
display(agg_table_t2m(df))"""
    ),
    md("## Daily PRECTOTCORR — side-by-side boxplots + summary table"),
    code(
        """fig = precip_boxplots(df)
plt.show()
display(agg_table_precip(df))"""
    ),
    md("## Extreme heat (T2M_MAX > 35°C) & consecutive dry days (< 1 mm/day)"),
    code(
        """heat = extreme_heat_days_by_year(df)
drought = max_dry_days_by_year_country(df)
display(heat.tail())
display(drought.tail())

fig = bar_by_country(heat, "heat_days_35", "Days per year with T2M_MAX > 35°C")
plt.show()
fig = bar_by_country(
    drought,
    "max_consecutive_dry_days",
    "Max consecutive dry days (PRECTOTCORR < 1 mm) per year",
)
plt.show()"""
    ),
    md("## Statistical test on T2M across countries"),
    code(
        """name, stat, pval, obj = anova_or_kruskal_t2m(df)
print(f"{name}: statistic={stat:.4g}, p-value={pval:.4g}")
if pval < 0.05:
    print("Small p-value: strong evidence that at least one country differs in mean rank / location.")
else:
    print("Large p-value: no strong evidence against equal distributions — still compare effect sizes.")"""
    ),
    md(
        "### Notes on testing\n\n"
        "**ANOVA** if roughly normal homogeneous groups; otherwise **Kruskal–Wallis** when variances diverge heavily."
    ),
    code(
        """rank_df = vulnerability_rank_summary(df, heat, drought)
display(rank_df)"""
    ),
    md(
        "## COP32 framing — five bullets (edit after running on official data)\n\n"
        "- **Which country warms fastest?** Compare slopes on the overlay — tie to anomalies vs a baseline if extended.\n"
        "- **Most unstable precipitation:** highest daily spread (std/boxplots) plus dry spell maxima.\n"
        "- **What extremes imply:** heat-day counts × dry spells show compound hazard for agriculture/water.\n"
        "- **Ethiopia vs neighbors:** read Ethiopia’s row vs peers in the tables and figures.\n"
        "- **Who to champion for finance:** pick the country with converging high stress in the ranking + narrative "
        "for **adaptation finance**, **loss & damage**, and **early warning** — justify with this table.\n"
    ),
]


def main() -> None:
    pairs = [
        ("ethiopia", "Ethiopia"),
        ("kenya", "Kenya"),
        ("sudan", "Sudan"),
        ("tanzania", "Tanzania"),
        ("nigeria", "Nigeria"),
    ]
    for slug, title in pairs:
        p = ROOT / "notebooks" / f"{slug}_eda.ipynb"
        write_ipynb(p, eda_notebook(slug, title))
        print("wrote", p)
    cp = ROOT / "notebooks" / "compare_countries.ipynb"
    write_ipynb(cp, COMPARE_CELLS)
    print("wrote", cp)


if __name__ == "__main__":
    main()
