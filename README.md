# Climate Challenge — Week 0 (EthioClimate / COP32)

Exploratory analysis of NASA POWER daily extracts for **Ethiopia, Kenya, Sudan, Tanzania, and Nigeria** (2015–2026 Challenge window).

**Do not commit CSVs.** Place official challenge files under `data/<country>.csv` and exported cleans under `data/<country>_clean.csv`. The `.gitignore` already excludes `data/` and `*.csv`.

## Prerequisites

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional Jupyter: `pip install jupyter`.

## Task 2 — Country EDA notebooks

Per the brief, work on branch `eda-<country>`. Notebooks are supplied for all five countries:

| Notebook | Input | Output |
|----------|-------|--------|
| `notebooks/ethiopia_eda.ipynb` | `data/ethiopia.csv` | `data/ethiopia_clean.csv` |
| `notebooks/kenya_eda.ipynb` | `data/kenya.csv` | `data/kenya_clean.csv` |
| `notebooks/sudan_eda.ipynb` | … | … |
| `notebooks/tanzania_eda.ipynb` | … | … |
| `notebooks/nigeria_eda.ipynb` | … | … |

Run cells top-to-bottom. Shared helpers: `notebooks/eda_pipeline.py`.

Regenerate notebooks from the template script after editing:

```bash
python scripts/generate_week0_notebooks.py
```

## Task 3 — Cross-country comparison

Notebook: `notebooks/compare_countries.ipynb` — requires **all five** `*_clean.csv` files. Helpers: `notebooks/compare_pipeline.py`. Suggested branch: `compare-countries`.

## Dashboard (Streamlit)

Layout: `app/main.py`, helpers `app/utils.py`. Features: country multi-select, year range slider, monthly mean trend for a selected variable, seaborn **boxplot** of daily `PRECTOTCORR` by country.

```bash
streamlit run app/main.py
```

Reads only `data/<country>_clean.csv`. For **Streamlit Community Cloud**, set Main file to `app/main.py`; provide data via Secrets or private storage—nothing under `data/` is in Git.

Optional: save a UI capture under `dashboard_screenshots/` for your PDF report (that folder is tracked; CSVs are not).

## CI

GitHub Actions installs `requirements.txt` and runs `python -m compileall` on tracked packages.

## References

NASA POWER, WMO *State of the Climate in Africa*, World Bank Climate Risk Profiles — cited in notebook markdown.
