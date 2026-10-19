#  _____ _____ ____   ____
# |  ___|  ___|  _ \ / ___|
# | |_  | |_  | |_) | |
# |  _| |  _| |  _ <| |___
# |_|   |_|   |_| \_\____|
# FFRC
"""现有COA首周：复现原pooled排序，再新算焦点历史剔除分解。"""
from pathlib import Path
import hashlib
import importlib.util
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
        if ((candidate / 'analysis/reservoir ecology 20261019').is_dir()
                and (candidate / 'analysis/reservoir innovation 20261019').is_dir()):
            return candidate
    raise FileNotFoundError('Required Reproducibility/analysis/reservoir ecology 20261019 not found among script ancestors')

REPRO = find_repro()
V28 = REPRO / 'analysis/reservoir innovation 20261019'
EXT = V28 / 'validation/external'
METRICS = ['familiar_coverage', 'heldout_coverage', 'shared_history_loss',
           'outside_other_history_loss', 'heldout_configuration_gain', 'retained_coverage',
           'net_loss', 'shared_covered_opportunity', 'outside_covered_opportunity',
           'outside_other_candidate']

def csv(z, name):
    z.to_csv(OUT / f'{name} 1010.csv', index=False, float_format='%.17g')

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def ratios(z):
    z = z.copy()
    z['gross_loss'] = z.shared_history_loss + z.outside_other_history_loss
    for name, numerator, denominator in [
            ('shared_gross_loss_share', 'shared_history_loss', 'gross_loss'),
            ('outside_gross_loss_share', 'outside_other_history_loss', 'gross_loss'),
            ('shared_opportunity_share', 'shared_covered_opportunity', 'familiar_coverage'),
            ('outside_opportunity_share', 'outside_covered_opportunity', 'familiar_coverage'),
            ('shared_loss_per_opportunity', 'shared_history_loss', 'shared_covered_opportunity'),
            ('outside_loss_per_opportunity', 'outside_other_history_loss', 'outside_covered_opportunity')]:
        z[name] = z[numerator] / z[denominator].replace(0, np.nan)
    return z

def main():
    paths = [V28 / 'public assets/NewmanSound_year1.csv', V28 / 'coverage_objectives_20261019.py',
             EXT / 'outputs/Allocations.csv', EXT / 'outputs/Fold support.csv',
             EXT / 'outputs/Fish day coverage.csv', EXT / 'outputs/Summary.csv', EXT / 'Protocol.md']
    spec = importlib.util.spec_from_file_location('existing_objective', paths[1])
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    d = pd.read_csv(paths[0]); d['time'] = pd.to_datetime(d.Hourbin, format='%d/%m/%Y %H:%M')
    d['fish'] = d.Tag.astype(str); d['cycle'] = d.time.dt.strftime('%Y-%m-%d')
    d['phase'] = 'All'; d['hour'] = d.time.dt.floor('h')
    d = d[d.Transplant.eq('Control')].copy()
    assert d.fish.nunique() == 21 and not d.duplicated(['fish', 'hour']).any()
    d['cell'] = np.floor(d.POINT_X / 100).astype(int).astype(str) + ':' + np.floor(d.POINT_Y / 100).astype(int).astype(str)
    counts = d.groupby(['fish', 'cycle']).hour.nunique().rename('hour_n')
    d = d.join(counts, on=['fish', 'cycle']); d = d[d.hour_n >= 12].copy()
    fold = pd.read_csv(paths[3]).query('fold == 0').iloc[0]
    train = d[(d.cycle >= fold.train_start) & (d.cycle < fold.test_start)]
    ids = sorted(train.groupby('fish').cycle.nunique().loc[lambda z: z >= 3].index.tolist())
    assert ids == sorted(fold.training_ids.split('|'))
    train = train[train.fish.isin(ids)]
    future = d[(d.cycle >= fold.test_start) & (d.cycle < fold.test_end_exclusive) & d.fish.isin(ids)]
    candidates = sorted(train.cell.unique())
    p, labels = m.probability_matrix(train, ids, candidates)
    chosen_a = [candidates[i] for i in m.top_cells(p.mean(axis=0), 8)]
    saved_alloc = pd.read_csv(paths[2]).query("fold == 0 and method == 'Pooled hotspot' and k == 8")
    assert chosen_a == saved_alloc.cell.tolist(), 'Original selected-cell ranking must reproduce exactly'
    rows = []; allocations = []; supports = []
    for fish, t in future.groupby('fish'):
        other_ids = [x for x in ids if x != fish]
        other = train[train.fish.isin(other_ids)]
        cells_b = sorted(other.cell.unique())
        pb, lab = m.probability_matrix(other, other_ids, cells_b)
        chosen_b = [cells_b[i] for i in m.top_cells(pb.mean(axis=0), 8)]
        assert set(cells_b).issubset(candidates) and len(chosen_b) == 8
        for rank, cell in enumerate(chosen_b, 1):
            allocations.append(dict(focal_fish=fish, rank=rank, k=8, cell=cell))
        supports.append(dict(focal_fish=fish, other_training_ids='|'.join(other_ids),
                             historical_fish_n=len(other_ids), other_candidate_cells=len(cells_b),
                             removed_candidate_cells='|'.join(sorted(set(candidates) - set(cells_b))),
                             selected_cells_changed=len(set(chosen_a) - set(chosen_b))))
        ca = t.cell.isin(chosen_a).to_numpy(); cb = t.cell.isin(chosen_b).to_numpy()
        inside = t.cell.isin(cells_b).to_numpy(); lost = ca & ~cb
        z = t[['fish', 'cycle', 'phase']].copy()
        z['familiar_coverage'] = ca.astype(float); z['heldout_coverage'] = cb.astype(float)
        z['shared_history_loss'] = (lost & inside).astype(float)
        z['outside_other_history_loss'] = (lost & ~inside).astype(float)
        z['heldout_configuration_gain'] = (~ca & cb).astype(float)
        z['retained_coverage'] = (ca & cb).astype(float)
        z['net_loss'] = ca.astype(float) - cb.astype(float)
        z['shared_covered_opportunity'] = (ca & inside).astype(float)
        z['outside_covered_opportunity'] = (ca & ~inside).astype(float)
        z['outside_other_candidate'] = (~inside).astype(float)
        q = z.groupby(['fish', 'cycle', 'phase'], as_index=False).agg(
            **{v: (v, 'mean') for v in METRICS}, hour_n=('familiar_coverage', 'size'))
        for v in ['shared_history_loss', 'outside_other_history_loss', 'shared_covered_opportunity', 'outside_covered_opportunity']:
            q[v + '_hours'] = np.rint(q[v] * q.hour_n).astype(int)
        rows.append(q)
    days = pd.concat(rows, ignore_index=True)
    saved = pd.read_csv(paths[4]).query("fold == 0 and method == 'Pooled hotspot' and k == 8")
    matched = days.merge(saved, on=['fish', 'cycle', 'phase'], suffixes=('', '_saved'), validate='one_to_one')
    assert len(days) == len(saved) == len(matched) == 127 and days.fish.nunique() == 20
    np.testing.assert_allclose(matched.familiar_coverage, matched.coverage, atol=1e-14, rtol=0)
    assert (matched.hour_n == matched.hour_n_saved).all()
    np.testing.assert_allclose(days.net_loss, days.shared_history_loss + days.outside_other_history_loss - days.heldout_configuration_gain, atol=1e-14, rtol=0)
    np.testing.assert_allclose(days.shared_covered_opportunity + days.outside_covered_opportunity, days.familiar_coverage, atol=1e-14, rtol=0)
    csv(days, 'Marine first week components per fish day')
    fish = days.groupby('fish', as_index=False).agg(**{v: (v, 'mean') for v in METRICS}, fish_days=('cycle', 'size'))
    csv(ratios(fish), 'Marine first week components per fish')
    summary = fish[METRICS].mean().to_frame().T
    summary['fish_n'] = len(fish); summary['fish_days'] = fish.fish_days.sum()
    summary = ratios(summary); csv(summary, 'Marine first week decomposition summary')
    original = pd.read_csv(paths[5]).query("scope == 'first_window' and method == 'Pooled hotspot'").iloc[0]
    assert abs(summary.familiar_coverage.iloc[0] - original.mean_coverage) < 1e-14
    csv(pd.DataFrame(allocations), 'Marine new excluded history configurations')
    csv(pd.DataFrame(supports), 'Marine new excluded history supports')
    s = summary.iloc[0].to_dict()
    clean = {key: (None if pd.isna(value) else value) for key, value in s.items()}
    audit = dict(status='PASS_NEW_CONDITIONAL_DECOMPOSITION',
                 summary=clean, original_first_week_mean=float(original.mean_coverage),
                 original_selected_cells_reproduced_in_rank_order=True,
                 original_fish_day_scores_reproduced=True, matched_fish_days=127,
                 historical_fish_n=len(ids), evaluation_fish_n=20,
                 original_candidate_cells=len(candidates), all_history_selected_cells=chosen_a,
                 excluded_configurations_evaluated=20,
                 configurations_with_any_selected_cell_change=sum(x['selected_cells_changed'] > 0 for x in supports),
                 fish_with_lost_future_coverage=int((fish.shared_history_loss + fish.outside_other_history_loss > 0).sum()),
                 source_dates=dict(train_start=fold.train_start, test_start=fold.test_start, test_end_exclusive=fold.test_end_exclusive),
                 weighting='Hourly COA coverage per qualified fish-day, equal days within fish, equal 20 evaluation fish; historical pooled ranking equal qualified days within fish then equal eligible historical fish.',
                 conditional_boundary='New first-week calculation using the existing protocol and pooled ranking, not a cached earlier decomposition. Marine hourly COA, All-day, unknown source timezone, >=12 hours/day, >=3 historical days/fish, 14-day history/7-day future, 100m grid, k=8. No maximin optimization, neural retraining, new threshold, or 53-week evaluation executed.',
                 zero_denominator_policy='Loss shares are undefined when gross loss is zero; regional loss/opportunity is undefined when that regional opportunity is zero. Null JSON and empty CSV fields report undefined quantities.',
                 source_files={str(p.relative_to(REPRO)): sha(p) for p in paths})
    (OUT / 'Marine executed decomposition audit 1010.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(audit, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
