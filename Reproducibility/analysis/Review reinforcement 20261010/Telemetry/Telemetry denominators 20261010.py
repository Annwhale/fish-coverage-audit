#  _____ _____ ____   ____
# |  ___|  ___|  _ \ / ___|
# | |_  | |_  | |_) | |
# |  _| |  _| |  _ <| |___
# |_|   |_|   |_| \_\____|
# FFRC
"""复算冻结选格：机会分母、描述性删鱼、个体和周的损失集中度。"""
from pathlib import Path
import hashlib
import json
import platform
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
BASE = REPRO / 'analysis'
V25 = BASE / 'reservoir ecology 20261019'
V28 = BASE / 'reservoir innovation 20261019'
V33 = BASE / 'innovation reframing 20261009'
METRICS = ['familiar_coverage', 'heldout_coverage', 'net_loss',
           'outside_other_history_loss', 'shared_history_loss',
           'heldout_configuration_gain', 'retained_coverage',
           'outside_familiar_candidate', 'outside_other_candidate',
           'shared_covered_opportunity', 'outside_covered_opportunity']
KEYS = ['method', 'k', 'phase']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def csv(z, name):
    z.to_csv(OUT / f'{name} 1010.csv', index=False, float_format='%.17g')

def ratios(z):
    z = z.copy()
    z['gross_loss'] = z.shared_history_loss + z.outside_other_history_loss
    z['shared_gross_loss_share'] = z.shared_history_loss / z.gross_loss
    z['outside_gross_loss_share'] = z.outside_other_history_loss / z.gross_loss
    z['shared_opportunity_share'] = z.shared_covered_opportunity / z.familiar_coverage
    z['outside_opportunity_share'] = z.outside_covered_opportunity / z.familiar_coverage
    z['shared_loss_per_opportunity'] = z.shared_history_loss / z.shared_covered_opportunity
    z['outside_loss_per_opportunity'] = z.outside_other_history_loss / z.outside_covered_opportunity
    return z

def primary(z):
    return z[(z.method == 'Pooled hotspot') & (z.k == 8) & (z.phase == 'Night')].copy()

def main():
    paths = {'positions': V25 / 'outputs/Primary sampled positions.parquet',
             'folds': V25 / 'outputs/Fold support primary.csv',
             'all_history_allocations': V25 / 'outputs/Training allocations.csv',
             'excluded_history_allocations': V28 / 'validation/outputs/Validation allocations.csv',
             'support_audits': V28 / 'validation/outputs/Support audits.csv',
             'saved_cycle_components': V33 / 'Transfer components per cycle.csv',
             'reviewed_decomposition_source': BASE / 'comprehensive audit 20261009/decompose transfer 20261009.py'}
    s = pd.read_parquet(paths['positions']); s['cycle'] = s.cycle.astype(str)
    folds = pd.read_csv(paths['folds'])
    a = pd.read_csv(paths['all_history_allocations'])
    b = pd.read_csv(paths['excluded_history_allocations'])
    audits = pd.read_csv(paths['support_audits'])
    rows = []; supports = []
    for f in folds.itertuples():
        train = s[(s.cycle >= f.train_start) & (s.cycle < f.test_start)]
        ids = f.training_fish.split('|')
        candidate_a = set(train[train.fish.isin(ids)].cell)
        test = s[(s.cycle >= f.test_start) & (s.cycle < f.test_end_exclusive) & s.fish.isin(ids)]
        paired = test.groupby(['fish', 'cycle']).phase.nunique().eq(2)
        test = test.join(paired.rename('paired'), on=['fish', 'cycle']).query('paired').drop(columns='paired')
        for fish, t in test.groupby('fish'):
            other = train[train.fish != fish]
            n = other[['fish', 'cycle', 'phase']].drop_duplicates().groupby(['fish', 'phase']).size().unstack(fill_value=0)
            other_ids = sorted(n.index[(n.Day >= 3) & (n.Night >= 3)].tolist())
            candidate_b = set(other[other.fish.isin(other_ids)].cell)
            saved = audits[(audits.design == 'B') & (audits.fold == f.fold) & (audits.heldout == fish)].iloc[0]
            assert set(saved.training_ids.split('|')) == set(other_ids)
            assert len(candidate_b) == saved.candidate_cells and candidate_b.issubset(candidate_a)
            supports.append(dict(fold=f.fold, fish=fish, other_training_ids='|'.join(other_ids),
                                 all_candidate_cells=len(candidate_a), other_candidate_cells=len(candidate_b)))
            in_other = t.cell.isin(candidate_b).to_numpy()
            for method in ['Day hotspot', 'Pooled hotspot', 'State balance']:
                for k in [1, 2, 4, 8, 16, 32]:
                    selected_a = set(a[(a.fold == f.fold) & (a.method == method) & (a.k == k)].cell)
                    selected_b = set(b[(b.design == 'B') & (b.fold == f.fold) & (b.heldout == fish) & (b.method == method) & (b.k == k)].cell)
                    assert len(selected_a) == len(selected_b) == k
                    assert selected_a.issubset(candidate_a) and selected_b.issubset(candidate_b)
                    ca = t.cell.isin(selected_a).to_numpy(); cb = t.cell.isin(selected_b).to_numpy()
                    loss = ca & ~cb
                    z = t[['fish', 'cycle', 'phase']].copy()
                    z['familiar_coverage'] = ca.astype(float); z['heldout_coverage'] = cb.astype(float)
                    z['net_loss'] = ca.astype(float) - cb.astype(float)
                    z['outside_other_history_loss'] = (loss & ~in_other).astype(float)
                    z['shared_history_loss'] = (loss & in_other).astype(float)
                    z['heldout_configuration_gain'] = (~ca & cb).astype(float)
                    z['retained_coverage'] = (ca & cb).astype(float)
                    z['outside_familiar_candidate'] = (~t.cell.isin(candidate_a)).to_numpy().astype(float)
                    z['outside_other_candidate'] = (~in_other).astype(float)
                    # 机会仅取原方案已覆盖的未来位置，两个区域完整分割原覆盖。
                    z['shared_covered_opportunity'] = (ca & in_other).astype(float)
                    z['outside_covered_opportunity'] = (ca & ~in_other).astype(float)
                    q = z.groupby(['fish', 'cycle', 'phase'], as_index=False)[METRICS].mean()
                    count = z.groupby(['fish', 'cycle', 'phase'], as_index=False).agg(sample_n=('familiar_coverage', 'size'))
                    q = q.merge(count, on=['fish', 'cycle', 'phase'], validate='one_to_one')
                    q['fold'] = f.fold; q['method'] = method; q['k'] = k
                    for name in ['shared_covered_opportunity', 'outside_covered_opportunity', 'shared_history_loss', 'outside_other_history_loss']:
                        q[name + '_positions'] = np.rint(q[name] * q.sample_n).astype(int)
                    rows.append(q)
    cycle = pd.concat(rows, ignore_index=True)
    original = pd.read_csv(paths['saved_cycle_components'])
    merge_keys = ['fold', 'fish', 'cycle', 'phase', 'method', 'k']
    matched = cycle.merge(original, on=merge_keys, suffixes=('', '_saved'), validate='one_to_one')
    assert len(matched) == len(cycle) == len(original) == 8244
    error = max(float(np.max(np.abs(matched[m] - matched[m + '_saved']))) for m in METRICS[:9])
    assert error < 1e-14
    np.testing.assert_allclose(cycle.shared_covered_opportunity + cycle.outside_covered_opportunity, cycle.familiar_coverage, atol=1e-14)
    np.testing.assert_allclose(cycle.outside_covered_opportunity, cycle.outside_other_history_loss, atol=1e-14)
    csv(cycle, 'Opportunity components per cycle')
    csv(pd.DataFrame(supports), 'Candidate support checks')
    fish = cycle.groupby(KEYS + ['fish'], as_index=False).agg(**{m: (m, 'mean') for m in METRICS}, paired_cycles=('cycle', 'size'))
    csv(ratios(fish), 'Opportunity components per fish')
    summary = fish.groupby(KEYS, as_index=False).agg(**{m: (m, 'mean') for m in METRICS}, fish_n=('fish', 'nunique'), paired_cycles=('paired_cycles', 'sum'))
    summary = ratios(summary); csv(summary, 'Opportunity summary')
    main = primary(summary).iloc[0].to_dict()
    assert main['fish_n'] == 5 and main['paired_cycles'] == 229
    # 删鱼只重新平均已保存的鱼级端点，选格、候选与未来位置均不重新拟合。
    deletions = []
    for omitted in sorted(fish.fish.unique()):
        z = fish[fish.fish != omitted]
        q = z.groupby(KEYS, as_index=False).agg(**{m: (m, 'mean') for m in METRICS}, fish_n=('fish', 'nunique'), paired_cycles=('paired_cycles', 'sum'))
        q = ratios(q); q['omitted_fish'] = omitted; deletions.append(q)
    deletion = pd.concat(deletions, ignore_index=True)
    csv(deletion, 'Descriptive delete one fish')
    p = primary(cycle); pf = primary(fish); nfish = pf.fish.nunique()
    pf = ratios(pf)
    for m in ['shared_history_loss', 'outside_other_history_loss', 'gross_loss', 'net_loss']:
        pf[m + '_weighted_contribution'] = pf[m] / nfish
        pf[m + '_contribution_share'] = pf[m] / pf[m].sum()
    csv(pf, 'Primary fish concentration')
    countfish = p.groupby('fish').cycle.size()
    p['cohort_weight'] = p.fish.map(1 / (nfish * countfish))
    p['gross_loss'] = p.shared_history_loss + p.outside_other_history_loss
    contribution_metrics = ['shared_history_loss', 'outside_other_history_loss', 'gross_loss', 'net_loss']
    for m in contribution_metrics:
        p[m + '_weighted_contribution'] = p[m] * p.cohort_weight
    week_contributions = p.groupby('fold', as_index=False).agg(
        **{m + '_weighted_contribution': (m + '_weighted_contribution', 'sum') for m in contribution_metrics},
        cohort_weight=('cohort_weight', 'sum'), fish_n=('fish', 'nunique'), paired_cycles=('cycle', 'size'))
    for m in contribution_metrics:
        week_contributions[m + '_contribution_share'] = week_contributions[m + '_weighted_contribution'] / week_contributions[m + '_weighted_contribution'].sum()
        np.testing.assert_allclose(week_contributions[m + '_weighted_contribution'].sum(), main[m], atol=1e-14)
    week_contributions = week_contributions.merge(folds[['fold', 'test_start', 'test_end_exclusive']], on='fold', validate='one_to_one')
    csv(week_contributions, 'Primary week contribution concentration')
    # 每周端点另按当周鱼内周期、当周鱼间等权计算；不是全季端点的权重。
    fishweek = p.groupby(['fish', 'fold'], as_index=False).agg(**{m: (m, 'mean') for m in METRICS}, paired_cycles=('cycle', 'size'))
    csv(ratios(fishweek), 'Primary fish week endpoints')
    week = fishweek.groupby('fold', as_index=False).agg(**{m: (m, 'mean') for m in METRICS}, fish_n=('fish', 'nunique'), paired_cycles=('paired_cycles', 'sum'))
    csv(ratios(week).merge(folds[['fold', 'test_start', 'test_end_exclusive']], on='fold'), 'Primary week equal fish endpoints')
    sens = []
    both = primary(cycle).drop(columns='phase')
    support = cycle[(cycle.method == 'Pooled hotspot') & (cycle.k == 8)].pivot(index=['fish', 'cycle'], columns='phase', values='sample_n')
    for threshold in [12, 24, 36, 48, 60]:
        eligible = support[(support.Day >= threshold) & (support.Night >= threshold)].index
        q = both.set_index(['fish', 'cycle']).loc[eligible].reset_index()
        f = q.groupby('fish')[METRICS].mean()
        row = f.mean().to_dict(); row.update(minimum_positions_both_windows=threshold, fish_n=len(f), paired_cycles=len(q))
        sens.append(row)
    csv(ratios(pd.DataFrame(sens)), 'Opportunity support sensitivity')
    external = V28 / 'validation/external'
    extpaths = [external / 'outputs/Allocations.csv', external / 'outputs/Fold support.csv',
                external / 'outputs/Fish day coverage.csv', external / 'run_external_20261019.py',
                external / 'outputs/Source and execution audit.json', V28 / 'public assets/NewmanSound_year1.csv']
    exta = pd.read_csv(extpaths[0]); extfold = pd.read_csv(extpaths[1])
    ext = dict(status='CACHE_HAS_NO_FOCAL_EXCLUDED_CONFIGURATIONS_NEW_CALCULATION_SEPARATE',
        saved_all_history_allocations_rows=len(exta), allocation_columns=list(exta.columns),
        saved_methods=sorted(exta.method.unique().tolist()), evaluable_folds=len(exta.fold.unique()),
        saved_folds=len(extfold), grid_m=100, budget=8,
        support='Hourly marine centres of activity (COA), all-day, source calendar time; >=12 hours/day, >=3 historical qualified days/fish, 14-day history and 7-day future.',
        missing='No focal-history-excluded selected-cell configurations or their candidate/support audits are saved. Pooled hotspot versus maximin is an objective comparison, not the A-versus-B focal-history exclusion comparison.',
        implication='The earlier cache alone cannot identify the focal-history-exclusion split. The separately authorized new first-week calculation is documented in Marine executed decomposition audit 1010.json and Marine conditional decomposition 20261010.py; no cached earlier replication is claimed.',
        files={str(x.relative_to(REPRO)): sha(x) for x in extpaths})
    (OUT / 'Marine decomposition readiness 1010.json').write_text(json.dumps(ext, ensure_ascii=False, indent=2) + '\n')
    concentration = {}
    for m in contribution_metrics:
        fs = pf[m + '_contribution_share']; ws = week_contributions[m + '_contribution_share']
        concentration[m] = dict(max_fish_share=float(fs.max()), max_fish=pf.loc[fs.idxmax(), 'fish'],
                                top_two_fish_share=float(fs.nlargest(2).sum()),
                                max_week_share=float(ws.max()), max_week_fold=int(week_contributions.loc[ws.idxmax(), 'fold']),
                                top_three_week_share=float(ws.nlargest(3).sum()))
    raw = dict(sampled_night_positions=int(p.sample_n.sum()),
               **{m: int(p[m + '_positions'].sum()) for m in ['shared_covered_opportunity', 'outside_covered_opportunity', 'shared_history_loss', 'outside_other_history_loss']})
    report = dict(status='PASS', primary=main, raw_counts_not_the_primary_weighting=raw,
        delete_one_fish_primary=primary(deletion).to_dict('records'), concentration=concentration,
        matches_saved_cycle_rows=len(cycle), maximum_saved_metric_difference=error,
        weighting='Per fish-cycle-window position fractions; equal cycles within each fish, then equal fish. Day and Night remain separate. Primary is Pooled hotspot, k=8, Night. Ratios are formed after averaging numerator and denominator with these identical weights.',
        week_contribution_weight='Each fish-cycle-window contributes 1/(5 * that fish total paired cycles). Week contribution sums reconstruct the full equal-fish seasonal summary; week endpoints separately use fish means within each week.',
        deletion_boundary='Descriptive reaggregation of frozen per-fish outputs after removing one evaluation fish; no re-fitting, no newly withheld history, no jackknife standard error or population inference.',
        opportunity_definition='All-history-covered future positions partitioned by membership in other eligible fish historical candidate cells. Shared means the retained other-history candidate support, not simultaneous co-occurrence or universal space use.',
        structural_constraint='Outside loss per covered opportunity equals 1 by construction because the excluded-history selected cells are a subset of the retained candidate support.',
        marine_cache=ext, marine_new_calculation_audit='Marine executed decomposition audit 1010.json',
        software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
        sources={key: dict(path=str(path.relative_to(REPRO)), sha256=sha(path)) for key, path in paths.items()})
    (OUT / 'Telemetry audit 1010.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(primary=main, deletion=report['delete_one_fish_primary'], concentration=concentration, raw=raw, marine_status=ext['status']), ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
