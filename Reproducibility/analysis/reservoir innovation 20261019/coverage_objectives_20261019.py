#  _____ _____ ____   ____
# |  ___|  ___|  _ \ / ___|
# | |_  | |_  | |_) | |
# |  _| |  _| |  _ <| |___
# |_|   |_|   |_| \_\____|
# FFRC
"""历史平均覆盖与最低组覆盖；不读取后期位置。"""
from pathlib import Path
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, hstack, vstack


def probability_matrix(train, fish_ids, cells):
    labels, probabilities = [], []
    phases = sorted(train.phase.unique())
    for fish in sorted(fish_ids):
        for phase in phases:
            z = train.loc[train.fish.eq(fish) & train.phase.eq(phase)]
            if z.empty:
                raise ValueError((fish, phase, 'missing historical group'))
            per_cycle = z.groupby('cycle').size()
            weight = 1 / z.cycle.map(per_cycle) / len(per_cycle)
            probability = weight.groupby(z.cell).sum().reindex(cells, fill_value=0).to_numpy()
            assert abs(probability.sum() - 1) < 1e-10
            labels.append((fish, phase))
            probabilities.append(probability)
    return np.asarray(probabilities), labels


def top_cells(probability, k):
    return np.lexsort((np.arange(len(probability)), -probability))[:k].tolist()


def solve_floor(probability, k, time_limit=30):
    groups, n = probability.shape
    if n < k:
        raise ValueError('insufficient historical candidate cells')
    # x为入选网格，t为所有历史组的最低覆盖。
    matrix = vstack([hstack([csr_matrix(probability), -np.ones((groups, 1))]),
                     csr_matrix(np.r_[np.ones(n), 0][None, :])], format='csr')
    lower = np.r_[np.zeros(groups), k]
    upper = np.r_[np.full(groups, np.inf), k]
    objective = np.r_[np.zeros(n), -1.]
    options = {'time_limit': time_limit, 'mip_rel_gap': 1e-7, 'threads': 1}
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected.*threads', category=RuntimeWarning)
        first = milp(objective, integrality=np.r_[np.ones(n), 0],
                     bounds=Bounds(np.zeros(n+1), np.ones(n+1)),
                     constraints=LinearConstraint(matrix, lower, upper), options=options)
    if first.x is None:
        raise RuntimeError(first.message)
    selected_first = np.flatnonzero(first.x[:n] > .5)
    assert len(selected_first) == k
    achieved_floor = float(probability[:, selected_first].sum(axis=1).min())
    tolerance = 1e-7
    # 保持最低覆盖，再以平均覆盖作次级目标；未来结果不参与破同分。
    matrix2 = vstack([csr_matrix(probability), csr_matrix(np.ones((1, n)))], format='csr')
    mean = probability.mean(axis=0)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected.*threads', category=RuntimeWarning)
        second = milp(-mean, integrality=np.ones(n), bounds=Bounds(np.zeros(n), np.ones(n)),
                      constraints=LinearConstraint(matrix2, np.r_[np.full(groups, achieved_floor-tolerance), k],
                                                   np.r_[np.full(groups, np.inf), k]), options=options)
    selected = np.flatnonzero(second.x > .5) if second.x is not None else selected_first
    assert len(selected) == k
    coverage = probability[:, selected].sum(axis=1)
    assert coverage.min() >= achieved_floor - 2e-6
    return selected.tolist(), {
        'stage1_status': int(first.status), 'stage1_message': first.message,
        'stage1_gap': float(first.mip_gap), 'stage1_dual_bound': float(first.mip_dual_bound),
        'stage2_status': int(second.status), 'stage2_message': second.message,
        'stage2_gap': float(second.mip_gap) if second.x is not None else None,
        'historical_minimum': float(coverage.min()), 'historical_mean': float(coverage.mean()),
        'stage1_achieved_floor': achieved_floor, 'lexicographic_tolerance': tolerance,
        'both_stages_optimal_to_solver_tolerance': first.status == 0 and second.status == 0,
        'candidate_cells': n, 'historical_groups': groups, 'k': k,
    }


def selections(probability, labels, k):
    methods = {'Pooled hotspot': top_cells(probability.mean(axis=0), k)}
    phases = [label[1] for label in labels]
    if set(phases) == {'Day', 'Night'}:
        day = probability[np.array(phases) == 'Day'].mean(axis=0)
        night = probability[np.array(phases) == 'Night'].mean(axis=0)
        methods['Day hotspot'] = top_cells(day, k)
        indices, available = [], np.ones(len(day), dtype=bool)
        day_total = night_total = 0.
        for _ in range(k):
            score = np.minimum(day_total+day, night_total+night)
            score[~available] = -np.inf
            chosen = int(np.lexsort((np.arange(len(day)), -(day+night)/2, -score))[0])
            indices.append(chosen)
            available[chosen] = False
            day_total += day[chosen]
            night_total += night[chosen]
        methods['State balance'] = indices
    chosen, audit = solve_floor(probability, k)
    methods['Individual window maximin'] = chosen
    return methods, audit
