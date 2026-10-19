![Welcome](assets/welcome.gif)

# Fish coverage audit

FFRC · Freshwater Fisheries Research Center

Code for *Locating individual-history coverage losses within shared space*.
Analyses cover individual coverage, shared-space losses and image-review budgets.

## Run

```bash
pip install -r requirements.txt
python "Reproducibility/analysis/reservoir innovation 20261019/application/verify_small_example_20261019.py"
python "Reproducibility/analysis/reservoir innovation 20261019/application/coverage_audit_20261019.py" coverage.csv --out results
```

Input columns: `unit`, `date`, `coverage` (0–1). Optional: `method`, `k`, `phase`.
The example uses synthetic data. Study analyses require the accompanying review data, arranged under the corresponding analysis directories.

## Data sources

- [Říha reservoir data](https://doi.org/10.5281/zenodo.19885546) · CC BY 4.0
- [Newman Sound data](https://doi.org/10.5061/dryad.2f2b3) · CC0

Original images and reports are not included. For code reuse, contact the authors.
