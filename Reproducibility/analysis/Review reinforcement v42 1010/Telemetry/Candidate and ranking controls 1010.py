#  _____ _____ ____   ____
# |  ___|  ___|  _ \ / ___|
# | |_  | |_  | |_) | |
# |  _| |  _| |  _ <| |___
# |_|   |_|   |_| \_\____|
# FFRC
"""主预算的候选与排序描述性对照；只读取已有位置和冻结选择。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent

def find_repro():
    parents = list(OUT.parents)
    # 优先真实Repro根，避免工作根另一个analysis缓存遮蔽投稿资料。
    candidates = ([p for p in parents if p.name == 'Reproducibility']
                  + [p / 'submission/Ecological Indicators/Reproducibility' for p in parents]
                  + parents)
    for candidate in candidates:
        if ((candidate / 'analysis/reservoir ecology v25').is_dir()
                and (candidate / 'analysis/reservoir innovation v28').is_dir()):
            return candidate
    raise FileNotFoundError('Required Reproducibility/analysis/reservoir ecology v25 not found among script ancestors')

REPRO = find_repro()
BASE = REPRO / 'analysis'
V25 = BASE / 'reservoir ecology v25'
V28 = BASE / 'reservoir innovation v28'
METRICS = ['coverage', 'loss_shared', 'loss_outside', 'gain', 'net_loss', 'retained']

def probability(d, ids, phase, cells):
    # 与v25 cell_probability同公式；保留鱼、周期、窗口的均衡。
    z = d[d.fish.isin(ids) & d.phase.eq(phase)].copy()
    per_cycle_n = z.groupby(['fish', 'cycle']).fish.transform('size')
    cycles = z[['fish', 'cycle']].drop_duplicates().groupby('fish').size()
    weights = 1 / per_cycle_n / z.fish.map(cycles) / len(ids)
    return weights.groupby(z.cell).sum().reindex(cells, fill_value=0).to_numpy()

def rank(p):
    return np.lexsort((np.arange(len(p)), -p))[:8]

def csv(z, name):
    z.to_csv(OUT / f'{name} 1010.csv', index=False, float_format='%.17g')

def main():
    paths = [V25 / 'outputs/Primary sampled positions.parquet', V25 / 'outputs/Fold support primary.csv',
             V25 / 'outputs/Training allocations.csv', V28 / 'validation/outputs/Validation allocations.csv',
             V25 / 'analyse_rimov.py', OUT / 'Opportunity components per cycle 1010.csv']
    s = pd.read_parquet(paths[0]); s['cycle'] = s.cycle.astype(str)
    folds = pd.read_csv(paths[1]); a = pd.read_csv(paths[2]); b = pd.read_csv(paths[3])
    records = []; alloc = []; checks = []
    for f in folds.itertuples():
        train = s[(s.cycle >= f.train_start) & (s.cycle < f.test_start)]
        ids_a = f.training_fish.split('|')
        cells_a = sorted(train[train.fish.isin(ids_a)].cell.unique())
        score_a = (probability(train, ids_a, 'Day', cells_a) + probability(train, ids_a, 'Night', cells_a)) / 2
        chosen_a = [cells_a[i] for i in rank(score_a)]
        saved_a = a[(a.fold == f.fold) & a.method.eq('Pooled hotspot') & (a.k == 8)].cell.tolist()
        assert chosen_a == saved_a
        future = s[(s.cycle >= f.test_start) & (s.cycle < f.test_end_exclusive) & s.fish.isin(ids_a)]
        paired = future.groupby(['fish', 'cycle']).phase.nunique().eq(2)
        future = future.join(paired.rename('paired'), on=['fish', 'cycle']).query('paired').drop(columns='paired')
        for fish, t in future.groupby('fish'):
            other = train[train.fish != fish]
            n = other[['fish', 'cycle', 'phase']].drop_duplicates().groupby(['fish', 'phase']).size().unstack(fill_value=0)
            ids_b = sorted(n.index[(n.Day >= 3) & (n.Night >= 3)].tolist())
            cells_b = sorted(other[other.fish.isin(ids_b)].cell.unique())
            score_b = (probability(other, ids_b, 'Day', cells_b) + probability(other, ids_b, 'Night', cells_b)) / 2
            chosen_b = [cells_b[i] for i in rank(score_b)]
            saved_b = b[b.design.eq('B') & (b.fold == f.fold) & (b.heldout == fish) & b.method.eq('Pooled hotspot') & (b.k == 8)].cell.tolist()
            assert chosen_b == saved_b
            restricted_score_a = pd.Series(score_a, index=cells_a).reindex(cells_b).to_numpy()
            chosen_c = [cells_b[i] for i in rank(restricted_score_a)]
            score_d = (probability(other, ids_b, 'Day', cells_a) + probability(other, ids_b, 'Night', cells_a)) / 2
            chosen_d = [cells_a[i] for i in rank(score_d)]
            assert chosen_d == chosen_b
            checks.append(dict(fold=f.fold, fish=fish, b_equals_d=True, b_c_changed_cells=len(set(chosen_b) - set(chosen_c)),
                               a_c_changed_cells=len(set(chosen_a) - set(chosen_c)), all_candidates=len(cells_a), other_candidates=len(cells_b)))
            in_other = t.cell.isin(cells_b).to_numpy(); ca = t.cell.isin(chosen_a).to_numpy()
            for label, selected in [('A', chosen_a), ('B', chosen_b), ('C', chosen_c), ('D', chosen_d)]:
                for pos, cell in enumerate(selected, 1): alloc.append(dict(fold=f.fold, fish=fish, control=label, rank=pos, cell=cell))
                covered = t.cell.isin(selected).to_numpy(); lost = ca & ~covered
                z = t[['fish', 'cycle', 'phase']].copy()
                z['coverage'] = covered.astype(float); z['loss_shared'] = (lost & in_other).astype(float)
                z['loss_outside'] = (lost & ~in_other).astype(float)
                z['gain'] = (~ca & covered).astype(float); z['net_loss'] = ca.astype(float) - covered.astype(float)
                z['retained'] = (ca & covered).astype(float)
                q = z.groupby(['fish', 'cycle', 'phase'], as_index=False)[METRICS].mean()
                q['fold'] = f.fold; q['control'] = label; records.append(q)
    cycle = pd.concat(records, ignore_index=True)
    saved = pd.read_csv(paths[5]); saved = saved[saved.method.eq('Pooled hotspot') & (saved.k == 8)]
    matched = cycle[cycle.control.eq('B')].merge(saved, on=['fish', 'cycle', 'phase', 'fold'], validate='one_to_one')
    assert len(matched) == 458
    for lhs, rhs in [('coverage', 'heldout_coverage'), ('loss_shared', 'shared_history_loss'), ('loss_outside', 'outside_other_history_loss'), ('gain', 'heldout_configuration_gain'), ('net_loss_x', 'net_loss_y')]:
        np.testing.assert_allclose(matched[lhs], matched[rhs], atol=1e-14, rtol=0)
    csv(cycle, 'Ranking control per cycle')
    fish = cycle.groupby(['control', 'phase', 'fish'], as_index=False).agg(**{m: (m, 'mean') for m in METRICS}, paired_cycles=('cycle', 'size'))
    csv(fish, 'Ranking control per fish')
    summary = fish.groupby(['control', 'phase'], as_index=False).agg(**{m: (m, 'mean') for m in METRICS}, fish_n=('fish', 'nunique'), paired_cycles=('paired_cycles', 'sum'))
    summary['gross_loss'] = summary.loss_shared + summary.loss_outside
    summary['shared_gross_loss_share'] = summary.loss_shared / summary.gross_loss.replace(0, np.nan)
    csv(summary, 'Ranking control summary')
    csv(pd.DataFrame(alloc), 'Ranking control configurations')
    csv(pd.DataFrame(checks), 'Ranking control configuration checks')
    audit = dict(status='PASS', controls=dict(A='All-history candidate support and all-history pooled score; saved original configuration reproduced.',
        B='Other eligible history candidate support and other-history pooled score; saved focal-history exclusion reproduced.',
        C='Other eligible history candidate support, retaining original all-history pooled score for ranking.',
        D='Original all-history candidate support, with only other-history pooled score; zero-score unsupported cells cannot enter top8 here.'),
        all_b_equals_d=True, fish_week_configurations=len(checks),
        c_preserves_all_original_selected_cells_inside_other_support=bool(cycle[cycle.control.eq('C')].loss_shared.eq(0).all()),
        budget=8, windows=['Day', 'Night'], weighting='Position fraction per fish-cycle-window, equal cycles within fish then equal five fish.',
        boundary='Descriptive sensitivity of candidate support and historical ranking under the same future observations; not ecological causal identification. No new ranking rule or threshold. C is an auxiliary accounting control, not proposed as an operational improvement.',
        night=[{key: (None if pd.isna(value) else value) for key, value in row.items()}
               for row in summary[summary.phase.eq('Night')].to_dict('records')],
        source_files={(str(p.relative_to(REPRO)) if p.is_relative_to(REPRO) else 'local outputs/' + p.name):
                      hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT / 'Ranking control audit 1010.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(audit, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
