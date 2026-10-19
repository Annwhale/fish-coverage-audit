![Welcome](assets/welcome.gif)

# Fish coverage audit

FFRC · Freshwater Fisheries Research Center

Code accompanying **Locating individual-history coverage losses within shared space**.

The analyses locate coverage lost when an individual’s history is excluded from a spatial monitoring design. They also compare mean coverage, repeated individual omissions and the spread of image review across acquisition sources.

## Quick check

```bash
python -m pip install -r requirements.txt
python "Reproducibility/analysis/reservoir innovation v28/application/verify_small_example.py"
```

This test uses three artificial individuals. It checks equal weighting, gaps between dates, missing coverage and duplicate records. It does not reproduce an empirical result.

## Audit a coverage table

```bash
python "Reproducibility/analysis/reservoir innovation v28/application/coverage_audit.py" coverage.csv --out results
```

Required columns: `unit` (or `fish`/`source`), `date` (or `cycle`), and `coverage` between zero and one. Missing dates or coverage break consecutive runs. Optional `method`, `k` and `phase` fields define comparison groups.

## Study analyses

The executed directory names are retained to match the review archive:

| Directory | Analysis |
|---|---|
| `reservoir ecology v25` | Historical selection and future reservoir coverage |
| `reservoir innovation v28` | Individual objectives, transfer and source coverage |
| `innovation reframing 20261009 v33` | Exact decomposition of coverage changes |
| `Review reinforcement v42 1010/Telemetry` | Opportunity denominators, ranking controls and marine comparison |
| `Hangbu SAT validation 20261008` | Evaluation of frozen underwater predictions |
| `Review reinforcement v42 1010/Imaging` | Annotation-status sensitivity |

For study replay, place the supplied analysis inputs at their recorded paths within `Reproducibility`. Run scripts from a working copy: several write output tables beside the inputs. The repository alone does not contain all inputs and cannot reproduce the full paper without the review archive. It does not include detector training, source images, model weights, satellite arrays or commissioned chemistry reports.

## Sources

Reservoir input: [Říha dataset](https://doi.org/10.5281/zenodo.19885546), CC BY 4.0. Marine input: [Newman Sound dataset](https://doi.org/10.5061/dryad.2f2b3), CC0. These identifiers refer to third-party data, not this code release. Obtain data from the source and retain the source attribution.

This release contains analysis code and a synthetic test only. Original imagery, annotation exports, internal paths and field reports are excluded. `source-manifest.json` records code-file checksums. No separate software reuse licence is assigned in this release; contact the authors for reuse permissions.

## Notes

实验参数和统计口径沿用冻结分析。注释以必要的中文说明为主；保留字段名、模型名和来源引用，便于与论文核对。FFRC 标识用于标明项目所属单位，不替代第三方资料的来源和许可。
