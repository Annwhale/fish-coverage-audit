#  _____ _____ ____   ____
# |  ___|  ___|  _ \ / ___|
# | |_  | |_  | |_) | |
# |  _| |  _| |  _ <| |___
# |_|   |_|   |_| \_\____|
# FFRC
"""按原配对队列检验未来个体遗漏和历史最低覆盖目标。"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse
import hashlib
import importlib.util
import json

import numpy as np
import pandas as pd

from coverage_objectives_20261019 import probability_matrix, selections

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'reservoir ecology 20261019'
OUT = ROOT / 'individual coverage'
OUT.mkdir(exist_ok=True)
K_VALUES = [1, 2, 4, 8, 16, 32]


def run_task(task):
    fold, k = task
    sampled = pd.read_parquet(BASE / 'outputs/Primary sampled positions.parquet')
    start = pd.Timestamp('2017-06-26') + pd.Timedelta(days=7*fold)
    cutoff, end = start + pd.Timedelta(days=14), start + pd.Timedelta(days=21)
    train = sampled.loc[sampled.cycle.ge(str(start.date())) & sampled.cycle.lt(str(cutoff.date()))].copy()
    support = train[['fish', 'cycle', 'phase']].drop_duplicates().groupby(['fish', 'phase']).size().unstack(fill_value=0)
    fish_ids = sorted(support.index[(support.Day >= 3) & (support.Night >= 3)].tolist())
    train = train.loc[train.fish.isin(fish_ids)]
    test = sampled.loc[sampled.cycle.ge(str(cutoff.date())) & sampled.cycle.lt(str(end.date())) & sampled.fish.isin(fish_ids)].copy()
    paired = test.groupby(['fish', 'cycle']).phase.nunique().eq(2)
    test = test.join(paired.rename('paired'), on=['fish', 'cycle'])
    test = test.loc[test.paired].copy()
    cells = sorted(train.cell.unique())
    probability, labels = probability_matrix(train, fish_ids, cells)
    methods, audit = selections(probability, labels, k)
    audit.update(fold=fold, training_start=str(start.date()), test_start=str(cutoff.date()),
                 test_end_exclusive=str(end.date()), training_fish=fish_ids)
    results, allocation = [], []
    for method, indices in methods.items():
        selected = [cells[i] for i in indices]
        z = test[['fish', 'cycle', 'phase']].copy()
        z['hit'] = test.cell.isin(selected).to_numpy()
        z['outside'] = ~test.cell.isin(cells).to_numpy()
        result = z.groupby(['fish', 'cycle', 'phase'], as_index=False).agg(
            coverage=('hit', 'mean'), sample_n=('hit', 'size'), outside_fraction=('outside', 'mean'))
        result['method'], result['k'], result['fold'] = method, k, fold
        results.append(result)
        covered = probability[:, indices].sum(axis=1)
        for (fish, phase), value in zip(labels, covered):
            allocation.append({'fold': fold, 'k': k, 'method': method, 'fish': fish,
                               'phase': phase, 'historical_coverage': value,
                               'selected_cells': '|'.join(selected)})
    return pd.concat(results), pd.DataFrame(allocation), audit


def longest_run(z, threshold):
    z = z.sort_values('cycle')
    longest = current = 0
    previous = None
    for row in z.itertuples():
        date = pd.Timestamp(row.cycle)
        low = row.coverage == 0 if threshold == 0 else row.coverage < threshold
        if low:
            current = current + 1 if previous is not None and date-previous == pd.Timedelta(days=1) else 1
            longest = max(longest, current)
        else:
            current = 0
        previous = date
    return longest


def summarize(result):
    per = result.groupby(['method', 'k', 'fish', 'phase'], as_index=False).agg(
        coverage=('coverage', 'mean'), cycle_n=('cycle', 'nunique'), test_weeks=('fold', 'nunique'),
        zero_fraction=('coverage', lambda x: float(x.eq(0).mean())),
        below10_fraction=('coverage', lambda x: float(x.lt(.10).mean())),
        below25_fraction=('coverage', lambda x: float(x.lt(.25).mean())),
        below50_fraction=('coverage', lambda x: float(x.lt(.50).mean())))
    per.to_csv(OUT / 'Per fish and window.csv', index=False)
    summary = per.groupby(['method', 'k', 'phase'], as_index=False).agg(
        mean_coverage=('coverage', 'mean'), minimum_fish_coverage=('coverage', 'min'),
        fish_n=('fish', 'nunique'), mean_zero_fraction=('zero_fraction', 'mean'),
        mean_below10_fraction=('below10_fraction', 'mean'))
    summary.to_csv(OUT / 'Mean and minimum coverage.csv', index=False)
    overall = per.groupby(['method', 'k'], as_index=False).agg(
        mean_coverage=('coverage', 'mean'), minimum_individual_window_coverage=('coverage', 'min'))
    overall.to_csv(OUT / 'Average and lower coverage frontier.csv', index=False)
    runs = []
    for key, z in result.groupby(['method', 'k', 'fish', 'phase']):
        for q in [0, .1, .25, .5]:
            runs.append(dict(zip(['method', 'k', 'fish', 'phase'], key), threshold=q,
                             longest_consecutive_observed_cycles=longest_run(z, q)))
    pd.DataFrame(runs).to_csv(OUT / 'Repeated omission runs.csv', index=False)
    spec = importlib.util.spec_from_file_location('baseline', BASE / 'analyse_rimov_20261019.py')
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    intervals = []
    for phase in ['Day', 'Night']:
        z = result.loc[result.k.eq(8) & result.phase.eq(phase)]
        interval = baseline.bootstrap_paired(z, 'Individual window maximin', 'Pooled hotspot')
        interval['phase'] = phase
        intervals.append(interval)
    pd.DataFrame(intervals).to_csv(OUT / 'Exploratory mean difference intervals.csv', index=False)
    primary = summary.loc[summary.k.eq(8)]
    print(primary.to_string(index=False), flush=True)
    print(pd.DataFrame(intervals).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        parts = list(pool.map(run_task, [(fold, k) for fold in range(11) for k in K_VALUES]))
    result = pd.concat([p[0] for p in parts], ignore_index=True)
    result.to_parquet(OUT / 'Future fish cycle coverage.parquet', index=False)
    pd.concat([p[1] for p in parts], ignore_index=True).to_csv(OUT / 'Historical configurations.csv', index=False)
    audits = [p[2] for p in parts]
    (OUT / 'Optimisation audit.json').write_text(json.dumps(audits, indent=2) + '\n')
    old = pd.read_parquet(BASE / 'outputs/Coverage primary.parquet').query('endpoint == "Position"')
    comparisons = result.query('method != "Individual window maximin"').merge(
        old[['fish', 'cycle', 'phase', 'method', 'k', 'fold', 'coverage']],
        on=['fish', 'cycle', 'phase', 'method', 'k', 'fold'], suffixes=('_new', '_original'), validate='one_to_one')
    assert len(comparisons) == len(result.query('method != "Individual window maximin"'))
    maximum = float((comparisons.coverage_new-comparisons.coverage_original).abs().max())
    assert maximum < 1e-12
    (OUT / 'Input and baseline verification.json').write_text(json.dumps({
        'status': 'PASS', 'original_baseline_maximum_difference': maximum,
        'matched_cycle_rows': len(comparisons),
        'input_sha256': hashlib.sha256((BASE / 'outputs/Primary sampled positions.parquet').read_bytes()).hexdigest(),
        'source_dataset_doi': '10.5281/zenodo.19885546',
        'all_optimisations_both_stages_optimal_to_solver_tolerance': all(x['both_stages_optimal_to_solver_tolerance'] for x in audits),
    }, indent=2) + '\n')
    summarize(result)
