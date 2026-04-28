# Scripts

## `generate_week0_notebooks.py`

Recreates Jupyter notebooks under `notebooks/` from structured templates (`*_eda.ipynb`, `compare_countries.ipynb`). Run from the repo root:

```bash
python scripts/generate_week0_notebooks.py
```

Use this when you tweak narrative cells or refactor shared helpers in `notebooks/eda_pipeline.py` / `compare_pipeline.py` and want consistent copies across all countries.

## `export_all_clean.py`

Writes `data/<country>_clean.csv` for Ethiopia–Nigeria (matches Task 2 notebook logic):

```bash
python scripts/export_all_clean.py
```
