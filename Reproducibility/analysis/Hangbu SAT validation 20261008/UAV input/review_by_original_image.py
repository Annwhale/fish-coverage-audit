#  _____ _____ ____   ____
# |  ___|  ___|  _ \ / ___|
# | |_  | |_  | |_) | |
# |  _| |  _| |  _ <| |___
# |_|   |_|   |_| \_\____|
# FFRC
"""原水库保存预测的按母图复核对照；不重新训练或调整检测阈值。"""
from pathlib import Path
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "analysis/annotation census 20261005/Input"
INPUT = HERE / "UAV inputs"
INPUT.mkdir(exist_ok=True)
for src, dest in [("visible_reference_iou030.csv", "paired_predictions.csv"),
                  ("merged_manifest.csv", "parent_metadata.csv")]:
    if not (INPUT / dest).exists():
        shutil.copy2(DATA / src, INPUT / dest)


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def p_none(n, m, k):
    if k > n - m:
        return 0.0
    return math.prod((n - k - j) / (n - j) for j in range(m))


# 用独立组合数公式核验并列候选抽取概率。
for n in range(1, 13):
    for m in range(n + 1):
        for k in range(n + 1):
            exact = 0.0 if k > n - m else math.comb(n - m, k) / math.comb(n, k)
            assert abs(p_none(n, m, k) - exact) < 1e-12

rows = read_csv(INPUT / "paired_predictions.csv")
meta = {r["original_filename"]: r["parent_stem"] for r in read_csv(INPUT / "parent_metadata.csv")}
parent = [meta[r["original_filename"]] for r in rows]
gt = [int(r["gt_count"]) > 0 for r in rows]
n_groups = len(set(parent))
positive_groups = len({p for p, hit in zip(parent, gt) if hit})
all_results = []
unit_results = []
for method in ["static", "bsca"]:
    scores = [int(r[f"{method}_pred_count"]) for r in rows]
    by_score = defaultdict(list)
    for i, score in enumerate(scores):
        by_score[score].append(i)
    global_curve = {}
    covered = set()
    prefix = 0
    for score in sorted(by_score, reverse=True):
        indexes = by_score[score]
        hits = Counter(parent[i] for i in indexes if gt[i])
        for q in range(1, len(indexes) + 1):
            k = prefix + q
            global_curve[k] = len(covered) + sum(1 - p_none(len(indexes), m, q)
                                                for p, m in hits.items() if p not in covered)
        covered.update(hits)
        prefix += len(indexes)
    group_indexes = defaultdict(list)
    for i, p in enumerate(parent):
        group_indexes[p].append(i)
    head_score_groups = defaultdict(list)
    for p, indexes in group_indexes.items():
        peak = max(scores[i] for i in indexes)
        heads = [i for i in indexes if scores[i] == peak]
        probability = sum(gt[i] for i in heads) / len(heads)
        head_score_groups[peak].append((p, probability))
    capped_curve = {}
    prefix, expected = 0, 0.0
    for score in sorted(head_score_groups, reverse=True):
        candidates = head_score_groups[score]
        total = sum(x[1] for x in candidates)
        for q in range(1, len(candidates) + 1):
            k = prefix + q
            capped_curve[k] = expected + q / len(candidates) * total
        expected += total
        prefix += len(candidates)
    assert abs(global_curve[len(rows)] - positive_groups) < 1e-9
    for k in range(1, len(rows) + 1):
        cutoff = sorted(scores, reverse=True)[k - 1]
        above = [i for i, value in enumerate(scores) if value > cutoff]
        tied = [i for i, value in enumerate(scores) if value == cutoff]
        positive_tiles = (sum(gt[i] for i in above) +
                          (k - len(above)) / len(tied) * sum(gt[i] for i in tied))
        unit_results.append({
            "method": method, "reviewed_tiles": k,
            "expected_positive_tiles": positive_tiles,
            "positive_tile_recall": positive_tiles / sum(gt),
            "expected_positive_parent_extents": global_curve[k],
            "positive_parent_extent_recall": global_curve[k] / positive_groups,
        })
    for k in range(1, n_groups + 1):
        all_results.append({
            "method": method, "reviewed_tiles": k,
            "count_ranking_expected_detected_positive_parent_extents": global_curve[k],
            "one_head_per_parent_expected_detected_positive_parent_extents": capped_curve[k],
            "group_minus_global": capped_curve[k] - global_curve[k],
            "positive_audited_parent_extents": positive_groups,
            "eligible_audited_parent_extents": n_groups,
        })
with (HERE / "UAV group review frontier.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(all_results[0]));w.writeheader();w.writerows(all_results)
with (HERE / "UAV review unit comparison.csv").open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(unit_results[0]));w.writeheader();w.writerows(unit_results)
summary = {
    "status": "PASS", "reference": "same 351 resolved UAV tiles; not complete-image fish absence",
    "prediction_source": "saved Static and BSCA counts; no retraining or retuning",
    "comparison_scope": "first review round, at most one highest-count tile from each audited parent extent",
    "review_positive_definition": "at least one reviewed reference-positive tile in the audited parent extent",
    "tie_rule": "uniform cutoff choice; uniform highest-score tile choice within each parent",
    "audited_tiles": len(rows), "audited_parents": n_groups,
    "reference_positive_parent_extents": positive_groups,
    "budget70": [r for r in all_results if r["reviewed_tiles"] == 70],
    "all_first_round_budgets": len(all_results),
    "method_summary": {},
    "field_visits_measured": False,
    "new_work_exploratory": True,
    "interpretation": "policy effect depends on endpoint and prediction errors; oracle gains do not establish achievable detector gains",
    "input_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in INPUT.iterdir() if p.is_file()},
}
by_method = {method: {r["reviewed_tiles"]: r for r in unit_results if r["method"] == method}
             for method in ["static", "bsca"]}
opposite = []
for k in range(1, len(rows) + 1):
    sr, br = by_method["static"][k], by_method["bsca"][k]
    tile_difference = br["positive_tile_recall"] - sr["positive_tile_recall"]
    parent_difference = br["positive_parent_extent_recall"] - sr["positive_parent_extent_recall"]
    if tile_difference * parent_difference < -1e-12:
        opposite.append(k)
summary["review_unit_comparison"] = {
    "all_budgets": len(rows),
    "budgets_with_opposite_effect_directions": opposite,
    "opposite_direction_budget_count": len(opposite),
    "budget70": [r for r in unit_results if r["reviewed_tiles"] == 70],
    "interpretation": "finite-cohort endpoint sensitivity; budgets are dependent, not independent experiments",
}
for method in ["static", "bsca"]:
    d = [r for r in all_results if r["method"] == method]
    summary["method_summary"][method] = {
        "budgets_improved": sum(r["group_minus_global"] > 1e-9 for r in d),
        "budgets_equal": sum(abs(r["group_minus_global"]) <= 1e-9 for r in d),
        "budgets_worse": sum(r["group_minus_global"] < -1e-9 for r in d),
        "maximum_gain": max(d, key=lambda r: r["group_minus_global"]),
        "maximum_loss": min(d, key=lambda r: r["group_minus_global"]),
    }
(HERE / "UAV group review summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
print(json.dumps({"budget70": summary["budget70"], "methods": summary["method_summary"]}, ensure_ascii=False))
