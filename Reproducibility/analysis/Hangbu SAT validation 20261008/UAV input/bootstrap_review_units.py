#  _____ _____ ____   ____
# |  ___|  ___|  _ \ / ___|
# | |_  | |_  | |_) | |
# |  _| |  _| |  _ <| |___
# |_|   |_|   |_| \_\____|
# FFRC
"""对同一批母图做配对重采样；区间只描述已选样本的条件不确定性。"""
from pathlib import Path
from collections import defaultdict
import csv
import json
import math

import numpy as np

HERE = Path(__file__).resolve().parent


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def p_none(n, m, k):
    if k > n - m:
        return 0.0
    return math.prod((n - k - j) / (n - j) for j in range(m))


def global_coverage(scores, groups, hits, k):
    cut = np.sort(scores)[-k]
    high = (scores > cut) & hits
    tie = scores == cut
    covered = set(groups[high].tolist())
    expected = float(len(covered))
    tie_hit_groups = groups[tie & hits]
    q = k - int((scores > cut).sum())
    n = int(tie.sum())
    if len(tie_hit_groups):
        units, counts = np.unique(tie_hit_groups, return_counts=True)
        expected += sum(1 - p_none(n, int(m), q) for unit, m in zip(units, counts) if unit not in covered)
    return expected


def grouped_coverage(scores, groups, hits, k):
    peaks, hit_probabilities = [], []
    for group in np.unique(groups):
        mask = groups == group
        peak = scores[mask].max()
        candidates = mask & (scores == peak)
        peaks.append(peak)
        hit_probabilities.append(hits[candidates].mean())
    peaks, hit_probabilities = np.array(peaks), np.array(hit_probabilities)
    cut = np.sort(peaks)[-k]
    high, tie = peaks > cut, peaks == cut
    q = k - int(high.sum())
    return float(hit_probabilities[high].sum() + q / int(tie.sum()) * hit_probabilities[tie].sum())


rows = read_csv(HERE / "UAV inputs/paired_predictions.csv")
meta = {r["original_filename"]: r["parent_stem"] for r in read_csv(HERE / "UAV inputs/parent_metadata.csv")}
ids = [meta[r["original_filename"]] for r in rows]
names = sorted(set(ids))
indices = [np.array([i for i, v in enumerate(ids) if v == name]) for name in names]
hit = np.array([int(r["gt_count"]) > 0 for r in rows])
static = np.array([int(r["static_pred_count"]) for r in rows])
bsca = np.array([int(r["bsca_pred_count"]) for r in rows])
rng = np.random.default_rng(20261008)
draws = 3000
k = 70
effects = []
for _ in range(draws):
    picked = rng.integers(0, len(names), size=len(names))
    idx = np.concatenate([indices[i] for i in picked])
    groups = np.concatenate([np.full(len(indices[j]), i) for i, j in enumerate(picked)])
    gt = hit[idx]
    denominator = len(np.unique(groups[gt]))
    if not denominator:
        continue
    sc, bc = static[idx], bsca[idx]
    sg = global_coverage(sc, groups, gt, k)
    bg = global_coverage(bc, groups, gt, k)
    sh = grouped_coverage(sc, groups, gt, k)
    bh = grouped_coverage(bc, groups, gt, k)
    # 同一母图重采样保持四个方法的配对；重复抽到的母图用抽样位置标识。
    effects.append([(bg - sg) / denominator, (sh - sg) / denominator, (bh - bg) / denominator])
values = np.array(effects)
point = json.loads((HERE / "UAV group review summary.json").read_text())["budget70"]
static_point = next(r for r in point if r["method"] == "static")
bsca_point = next(r for r in point if r["method"] == "bsca")
estimates = [
    (bsca_point["count_ranking_expected_detected_positive_parent_extents"] -
     static_point["count_ranking_expected_detected_positive_parent_extents"]) / 20,
    static_point["group_minus_global"] / 20,
    bsca_point["group_minus_global"] / 20,
]
labels = ["BSCA_minus_Static_global_parent_recall", "group_minus_global_Static_parent_recall",
          "group_minus_global_BSCA_parent_recall"]
out = {
    "draws_requested": draws, "draws_valid": len(effects), "seed": 20261008,
    "resampling_unit": "original image; all audited tiles retained in each sampled image",
    "budget_tiles": k, "fixed_number_of_sampled_original_images": 81,
    "interval": "paired percentile 2.5/97.5%; positive-parent denominator varies with resampled cohort",
    "bounds": {},
    "sampling_probability_known": False,
    "independent_flights_or_sites_generated": False,
    "scope": "conditional uncertainty for the selected retrospective cohort",
}
for i, label in enumerate(labels):
    low, high = np.quantile(values[:, i], [.025, .975])
    out["bounds"][label] = {"estimate": estimates[i], "low": float(low), "high": float(high)}
(HERE / "UAV group review uncertainty.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(out, ensure_ascii=False))
