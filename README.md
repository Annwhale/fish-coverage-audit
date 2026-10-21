![Welcome](assets/welcome.gif)

# Fish coverage audit

FFRC · Freshwater Fisheries Research Center

Code for *Diagnosing coverage losses in aquatic monitoring under limited effort*.

## Run

```bash
pip install -r requirements.txt
python analysis_20261021/allocation_20261021.py --inputs decision/Analysis/Inputs --out results/allocation
python analysis_20261021/ceiling_20261021.py --inputs evaluation/Analysis/Inputs --prior evaluation/Analysis/Prior --out results/diagnosis
python analysis_20261021/history_20261021.py --inputs additional/data --out results/history
```

Extract `Decision analyses_20261021.zip`, `Evaluation trial_20261021.zip` and
`Additional analyses_20261021.zip` from the manuscript supplements into
`decision`, `evaluation` and `additional`, respectively.

Earlier image and telemetry analyses remain in `Reproducibility/analysis`.

## Sources

- [Říha reservoir data](https://doi.org/10.5281/zenodo.19885546) · CC BY 4.0
- [Newman Sound data](https://doi.org/10.5061/dryad.2f2b3) · CC0

Original images and commissioned reports are not included.
For code reuse or data requests: duanjinrong@ffrc.cn.
