"""EthioClimate — Week 0 Streamlit dashboard (reads local CSVs under `data/`, ignored by git)."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.utils import (  # noqa: E402
    COUNTRY_SLUGS,
    filter_year_range,
    load_cleaned_dataframe,
    monthly_mean_t2m,
    pivot_variable_monthly_mean,
)


def discover_year_bounds() -> tuple[int, int]:
    """Infer global min/max YEAR from whichever clean exports exist locally."""
    years: list[int] = []
    for slug in COUNTRY_SLUGS.values():
        p = ROOT / "data" / f"{slug}_clean.csv"
        if not p.exists():
            continue
        y = pd.read_csv(p, usecols=["YEAR"], nrows=50000)
        years.extend([int(y["YEAR"].min()), int(y["YEAR"].max())])
    if not years:
        return 2015, 2026
    return min(years), max(years)


st.set_page_config(page_title="EthioClimate COP32 explorer", layout="wide")
st.title("EthioClimate Analytics — COP32 explorer")
st.caption(
    "Interactive views over NASA POWER–style cleaned daily CSVs (`data/<country>_clean.csv`). Raw data is never committed."
)

countries_all = list(COUNTRY_SLUGS.keys())
selected = st.multiselect(
    "Countries",
    countries_all,
    default=["Ethiopia", "Kenya"],
    help="Multi-select filters all charts.",
)
variable = st.selectbox(
    "Variable for monthly trend",
    ["T2M", "PRECTOTCORR", "RH2M", "WS2M", "T2M_MAX", "T2M_MIN"],
    index=0,
)

global_y0, global_y1 = discover_year_bounds()
year_range = st.slider("Year range", min_value=global_y0, max_value=global_y1, value=(global_y0, global_y1))

if not selected:
    st.warning("Pick at least one country.")
    st.stop()

try:
    dfall = load_cleaned_dataframe(selected)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

dff = filter_year_range(dfall, year_range[0], year_range[1])

left, right = st.columns([1.05, 0.95])
with left:
    st.subheader("Monthly mean — trend")
    if variable == "T2M":
        series = monthly_mean_t2m(dff).rename(columns={"T2M": "_v"})
    else:
        pm = pivot_variable_monthly_mean(dff, variable)
        series = pm.rename(columns={variable: "_v"})
    if series.empty:
        st.info("No rows in selection for plotting.")
    else:
        chart = (
            series.pivot(index="date", columns="Country", values="_v")
            .sort_index()
            .interpolate(limit_direction="both")
        )
        st.line_chart(chart, height=340)

with right:
    st.subheader("Daily PRECTOTCORR — boxplots")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    order = sorted(dff["Country"].unique())
    sns.boxplot(data=dff, x="Country", y="PRECTOTCORR", order=order, ax=ax)
    ax.set_ylabel("mm / day")
    ax.tick_params(axis="x", rotation=18)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.markdown("---")
st.caption(
    "Deploy: set `main` to `app/main.py` on Streamlit Community Cloud; supply data via private storage or "
    "mount—this repo intentionally omits CSVs."
)
