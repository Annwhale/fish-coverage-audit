#  _____ _____ ____   ____
# |  ___|  ___|  _ \ / ___|
# | |_  | |_  | |_) | |
# |  _| |  _| |  _ <| |___
# |_|   |_|   |_| \_\____|
# FFRC
from pathlib import Path
import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / 'alternative assets/zenodo_rimov_19885546/raw_positions.csv'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
K_VALUES = (1, 2, 4, 8, 16, 32)
SEED = 20261008
EXPECTED_BYTES = 386848790
EXPECTED_MD5 = 'fc105e9e18e99dc65b065448f520ad77'


def file_digest(path, algorithm):
    digest = hashlib.new(algorithm)
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(part)
    return digest.hexdigest()


def prepare():
    # 完整文件校验在解析前执行，不能把预分配下载当成完整数据。
    md5 = file_digest(RAW, 'md5')
    if RAW.stat().st_size != EXPECTED_BYTES or md5 != EXPECTED_MD5:
        raise ValueError(f'Incomplete/source-mismatched file: {RAW.stat().st_size}, {md5}')
    d = pd.read_csv(RAW)
    d['time'] = pd.to_datetime(d['up_timestamp_utc'], utc=True, format='mixed')
    d['fish'] = d['fi_fishid'].astype(str)
    d['local_time'] = d['time'].dt.tz_convert('Europe/Prague')
    d['cycle'] = (d['local_time'] - pd.Timedelta(hours=4)).dt.date.astype(str)
    hour = d['local_time'].dt.hour
    d['phase'] = np.select([(hour >= 8) & (hour < 16), (hour >= 22) | (hour < 4)], ['Day', 'Night'], default='Transition')
    residual_check = np.hypot(d['up_easting'] - d['up_egam'], d['up_northing'] - d['up_ngam'])
    d['base_valid'] = (d['up_validpos'].eq(True) & d['up_fishvalid'].eq(True)
                       & d[['up_easting', 'up_northing', 'time']].notna().all(axis=1)
                       & d['fi_fishid'].notna())
    registry = d.groupby('fish', observed=True).agg(
        source_rows=('fish', 'size'), first_time=('time', 'min'), last_time=('time', 'max'),
        marked_valid=('base_valid', 'sum'), depth_nonmissing=('up_depth', 'count'))
    season = d['cycle'].between('2017-06-26', '2017-09-30')
    for cut in (25, 50, 75):
        counts = d.loc[season & d['base_valid'] & (d['up_gamres'] < cut)].groupby('fish').size()
        registry[f'season_valid_residual_lt_{cut}'] = counts.reindex(registry.index, fill_value=0)
    registry.to_csv(OUT / 'Source fish registry.csv')
    # 同时核经纬度和米坐标；公开表保留了两套坐标，主分析不用GAM坐标。
    # 数值坐标核查用本机正常的系统PROJ；科学分析与绘图仍在同一Python环境运行。
    xy = d[['up_easting', 'up_northing']].iloc[::1000].to_numpy().tolist()
    conversion = "import json,sys; from pyproj import Transformer; xy=json.load(sys.stdin); t=Transformer.from_crs(32633,4326,always_xy=True); print(json.dumps([t.transform(*p) for p in xy]))"
    projection_environment = {k: v for k, v in os.environ.items() if k not in ['PROJ_DATA', 'PROJ_LIB']}
    result = subprocess.run([sys.executable, '-c', conversion], input=json.dumps(xy), capture_output=True, text=True, check=True, env=projection_environment)
    lonlat = np.asarray(json.loads(result.stdout))
    lon, lat = lonlat[:, 0], lonlat[:, 1]
    ll_error = np.hypot(np.asarray(lon) - d['lon'].to_numpy()[::1000], np.asarray(lat) - d['lat'].to_numpy()[::1000])
    audit = {
        'file_bytes': RAW.stat().st_size, 'md5': md5, 'sha256': file_digest(RAW, 'sha256'),
        'source_rows': len(d), 'source_fish_ids': registry.index.tolist(),
        'first_utc': str(d['time'].min()), 'last_utc': str(d['time'].max()),
        'marked_valid_rows': int(d['base_valid'].sum()),
        'missing_depth_fraction': float(d['up_depth'].isna().mean()),
        'raw_to_gam_residual_max_difference_m': float(np.nanmax(np.abs(residual_check - d['up_gamres']))),
        'epsg32633_lonlat_error_max_degrees': float(np.nanmax(ll_error)),
        'duplicate_fish_timestamps': int(d.duplicated(['fish', 'time']).sum()),
        'publisher_marks_not_independently_revalidated': ['up_fishvalid', 'up_validpos'],
        'temperature_column_present': False,
    }
    (OUT / 'Source audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    clean = d.loc[season & d['base_valid'] & d['phase'].isin(['Day', 'Night'])].copy()
    clean['bucket'] = clean['time'].astype('int64') // (300 * 10**9)
    clean['central_distance'] = np.abs(clean['time'].astype('int64') - (clean['bucket'] * 300 + 150) * 10**9)
    clean = clean.sort_values(['fish', 'time', 'up_gamres']).drop_duplicates(['fish', 'time'])
    keep = ['fish', 'time', 'cycle', 'phase', 'up_easting', 'up_northing', 'up_egam', 'up_ngam',
            'up_depth', 'up_bottom_depth', 'up_gamres', 'bucket', 'central_distance']
    clean[keep].to_parquet(OUT / 'Season valid window positions.parquet', index=False)
    print(json.dumps(audit), flush=True)


def sample_rows(raw, residual, sample_minutes, min_samples, coordinate):
    d = raw.loc[raw['up_gamres'] < residual].copy()
    nanos = sample_minutes * 60 * 10**9
    d['bucket'] = d['time'].astype('int64') // nanos
    d['central_distance'] = np.abs(d['time'].astype('int64') - (d['bucket'] + 0.5) * nanos)
    d = d.sort_values(['fish', 'bucket', 'central_distance', 'time']).drop_duplicates(['fish', 'bucket'])
    d['x'] = d['up_easting'] if coordinate == 'raw' else d['up_egam']
    d['y'] = d['up_northing'] if coordinate == 'raw' else d['up_ngam']
    d = d.sort_values(['fish', 'time'])
    delta_time = d.groupby('fish')['time'].diff().dt.total_seconds()
    distance = np.hypot(d.groupby('fish')['x'].diff(), d.groupby('fish')['y'].diff())
    d['move_50'] = (delta_time > 0) & (delta_time <= 600) & (distance >= 50)
    d['move_100'] = (delta_time > 0) & (delta_time <= 600) & (distance >= 100)
    d['move_200'] = (delta_time > 0) & (delta_time <= 600) & (distance >= 200)
    counts = d.groupby(['fish', 'cycle', 'phase']).size().rename('sample_count')
    d = d.join(counts, on=['fish', 'cycle', 'phase'])
    return d.loc[d['sample_count'] >= min_samples].copy(), counts


def cell_probability(d, fish_ids, phase, cells):
    z = d.loc[d['fish'].isin(fish_ids) & d['phase'].eq(phase)].copy()
    if z.empty:
        return np.zeros(len(cells))
    per_cycle_n = z.groupby(['fish', 'cycle'])['fish'].transform('size')
    cycle_per_fish = z[['fish', 'cycle']].drop_duplicates().groupby('fish').size()
    weights = 1 / per_cycle_n / z['fish'].map(cycle_per_fish) / len(fish_ids)
    probs = weights.groupby(z['cell']).sum()
    return probs.reindex(cells, fill_value=0).to_numpy()


def stable_rank(p):
    return np.lexsort((np.arange(len(p)), -p)).tolist()


def configuration_orders(day, night, centers):
    pooled = (day + night) / 2
    orders = {'Day hotspot': stable_rank(day), 'Night hotspot': stable_rank(night),
              'Pooled hotspot': stable_rank(pooled)}
    # 普通空间平衡对照；候选仅来自训练，不用未来出现位置。
    first = int(np.argmin(np.square(centers - centers.mean(axis=0)).sum(axis=1)))
    spatial = [first]
    mind = np.square(centers - centers[first]).sum(axis=1)
    for _ in range(min(32, len(day)) - 1):
        mind[spatial] = -1
        selected = int(np.argmax(mind))
        spatial.append(selected)
        mind = np.minimum(mind, np.square(centers - centers[selected]).sum(axis=1))
    orders['Spatial balance'] = spatial
    balanced = []
    available = np.ones(len(day), dtype=bool)
    day_total = night_total = 0.
    for _ in range(min(32, len(day))):
        objective = np.minimum(day_total + day, night_total + night)
        objective[~available] = -np.inf
        selected = int(np.lexsort((np.arange(len(day)), -pooled, -objective))[0])
        balanced.append(selected)
        day_total += day[selected]
        night_total += night[selected]
        available[selected] = False
    orders['State balance'] = balanced
    return orders


def quota_selection(day, night, k, day_first=True):
    day_order, night_order = stable_rank(day), stable_rank(night)
    first, second = (day_order, night_order) if day_first else (night_order, day_order)
    selected = first[: (k + 1) // 2]
    if len(selected) >= k:
        return selected
    for cell in second:
        if cell not in selected:
            selected.append(cell)
        if len(selected) >= k:
            break
    return selected


def fish_cycle_coverage(test, selected, method, k, fold, config_name, seed=-1):
    hit = test['cell'].isin(selected)
    cols = ['fish', 'cycle', 'phase']
    values = test[cols].copy()
    values['coverage'] = hit.to_numpy()
    values['n'] = 1
    parts = []
    for endpoint, mask in [('Position', np.ones(len(test), dtype=bool)),
                           ('Move50', test['move_50'].to_numpy()), ('Move100', test['move_100'].to_numpy()),
                           ('Move200', test['move_200'].to_numpy()),
                           ('Depth0to2', test['up_depth'].between(0, 2).to_numpy()),
                           ('Depth2to5', ((test['up_depth'] > 2) & (test['up_depth'] <= 5)).to_numpy()),
                           ('DepthOver5', (test['up_depth'] > 5).to_numpy())]:
        if not np.any(mask):
            continue
        part = values.loc[mask].groupby(cols, as_index=False).agg(coverage=('coverage', 'mean'), sample_n=('n', 'sum'))
        part['endpoint'] = endpoint
        parts.append(part)
    r = pd.concat(parts, ignore_index=True)
    r['method'], r['k'], r['fold'], r['configuration'], r['seed'] = method, k, fold, config_name, seed
    return r


def run_configuration(args):
    name, grid_size, residual, minutes, minimum, coordinate = args[:6]
    offset_x, offset_y = args[6:] if len(args) == 8 else (0, 0)
    raw = pd.read_parquet(OUT / 'Season valid window positions.parquet')
    sampled, counts = sample_rows(raw, residual, minutes, minimum, coordinate)
    sampled['cellx'] = np.floor((sampled['x'] - offset_x) / grid_size).astype(int)
    sampled['celly'] = np.floor((sampled['y'] - offset_y) / grid_size).astype(int)
    sampled['cell'] = sampled['cellx'].astype(str) + ':' + sampled['celly'].astype(str)
    native_x = raw['up_easting'] if coordinate == 'raw' else raw['up_egam']
    native_y = raw['up_northing'] if coordinate == 'raw' else raw['up_ngam']
    raw['cell'] = np.floor((native_x - offset_x) / grid_size).astype(int).astype(str) + ':' + np.floor((native_y - offset_y) / grid_size).astype(int).astype(str)
    if name == 'primary':
        sampled.to_parquet(OUT / 'Primary sampled positions.parquet', index=False)
        counts.to_csv(OUT / 'Fish cycle phase sampling counts.csv')
    rows, allocations, audit, random_distribution = [], [], [], []
    first_date = pd.Timestamp('2017-06-26')
    last_date = pd.Timestamp('2017-09-30')
    fold = 0
    start = first_date
    while start + pd.Timedelta(days=21) <= last_date + pd.Timedelta(days=1):
        cutoff = start + pd.Timedelta(days=14)
        end = cutoff + pd.Timedelta(days=7)
        train = sampled.loc[sampled['cycle'].ge(str(start.date())) & sampled['cycle'].lt(str(cutoff.date()))]
        test = sampled.loc[sampled['cycle'].ge(str(cutoff.date())) & sampled['cycle'].lt(str(end.date()))]
        n_cycles = train[['fish', 'cycle', 'phase']].drop_duplicates().groupby(['fish', 'phase']).size().unstack(fill_value=0)
        if not {'Day', 'Night'}.issubset(n_cycles.columns):
            start += pd.Timedelta(days=7)
            fold += 1
            continue
        train_fish = n_cycles.index[(n_cycles['Day'] >= 3) & (n_cycles['Night'] >= 3)].tolist()
        paired = test.groupby(['fish', 'cycle'])['phase'].nunique().eq(2)
        test = test.join(paired.rename('paired'), on=['fish', 'cycle'])
        unpaired_test_n = len(test)
        test = test.loc[test['paired'] & test['fish'].isin(train_fish)].copy()
        fish_ids = sorted(test['fish'].unique())
        train = train.loc[train['fish'].isin(train_fish)].copy()
        if not fish_ids or train.empty:
            start += pd.Timedelta(days=7)
            fold += 1
            continue
        # 布点学习使用训练合格鱼，不根据未来个体/响应重新学习。
        cells = sorted(train['cell'].unique())
        center = train[['cell', 'cellx', 'celly']].drop_duplicates('cell').set_index('cell').reindex(cells)
        centers = (center[['cellx', 'celly']].to_numpy() + 0.5) * grid_size + np.array([offset_x, offset_y])
        day = cell_probability(train, train_fish, 'Day', cells)
        night = cell_probability(train, train_fish, 'Night', cells)
        orders = configuration_orders(day, night, centers)
        raw_training = raw.loc[raw['cycle'].ge(str(start.date())) & raw['cycle'].lt(str(cutoff.date()))
                               & raw['phase'].eq('Day') & raw['fish'].isin(train_fish) & (raw['up_gamres'] < residual)]
        raw_count = raw_training.groupby('cell').size().reindex(cells, fill_value=0).to_numpy()
        orders['Day raw count'] = stable_rank(raw_count)
        test_outside = float((~test['cell'].isin(cells)).mean())
        audit.append({'configuration': name, 'fold': fold, 'train_start': str(start.date()),
                      'test_start': str(cutoff.date()), 'test_end_exclusive': str(end.date()),
                      'training_fish': '|'.join(train_fish), 'testing_fish': '|'.join(fish_ids),
                      'training_points': len(train), 'paired_testing_points': len(test),
                      'all_eligible_testing_points': unpaired_test_n, 'candidate_cells': len(cells),
                      'test_points_outside_training_cells': test_outside})
        rng = np.random.default_rng(SEED + fold)
        permutations = [rng.permutation(len(cells)) for _ in range(100)]
        for k in K_VALUES:
            if len(cells) < k:
                continue
            method_selections = {method: order[:k] for method, order in orders.items()}
            method_selections['Phase quota'] = quota_selection(day, night, k, day_first=fold % 2 == 0)
            for method, indices in method_selections.items():
                assert len(indices) == k and len(set(indices)) == k, (method, k, indices)
                selected = [cells[i] for i in indices]
                rows.append(fish_cycle_coverage(test, selected, method, k, fold, name))
                if name == 'primary':
                    for i in indices:
                        allocations.append({'fold': fold, 'test_start': str(cutoff.date()), 'method': method,
                                            'k': k, 'cell': cells[i], 'x': centers[i, 0], 'y': centers[i, 1],
                                            'training_day_probability': day[i], 'training_night_probability': night[i]})
            # 随机重复仅描述同成本配置分布，先在同鱼周期内平均，不增大生物n。
            random_probability = np.zeros(len(cells))
            for order in permutations:
                random_probability[order[:k]] += 1 / len(permutations)
            if name == 'primary':
                for seed, order in enumerate(permutations):
                    selected_set = set(cells[i] for i in order[:k])
                    z = test[['fish', 'cycle', 'phase']].copy()
                    z['coverage'] = test['cell'].isin(selected_set).to_numpy()
                    z = z.groupby(['fish', 'cycle', 'phase'], as_index=False)['coverage'].mean()
                    for phase in ['Day', 'Night']:
                        value = z.loc[z['phase'] == phase].groupby('fish')['coverage'].mean().mean()
                        random_distribution.append({'fold': fold, 'seed': seed, 'k': k, 'phase': phase, 'coverage': value})
            # 先求各单元100个配置中的入选频率；对测试位置取平均等价于逐配置求覆盖后平均。
            random_test = test.copy()
            random_test['_random_probability'] = random_test['cell'].map(dict(zip(cells, random_probability))).fillna(0)
            random_result = fish_cycle_coverage(random_test, [], 'Random', k, fold, name)
            endpoint_masks = {'Position': np.ones(len(test), dtype=bool),
                              'Move50': test['move_50'].to_numpy(), 'Move100': test['move_100'].to_numpy(),
                              'Move200': test['move_200'].to_numpy(),
                              'Depth0to2': test['up_depth'].between(0, 2).to_numpy(),
                              'Depth2to5': ((test['up_depth'] > 2) & (test['up_depth'] <= 5)).to_numpy(),
                              'DepthOver5': (test['up_depth'] > 5).to_numpy()}
            grouped_probs = []
            for endpoint, mask in endpoint_masks.items():
                z = random_test.loc[mask].groupby(['fish', 'cycle', 'phase'], as_index=False).agg(coverage=('_random_probability', 'mean'))
                z['endpoint'] = endpoint
                grouped_probs.append(z)
            probabilities = pd.concat(grouped_probs, ignore_index=True)
            random_result = random_result.drop(columns='coverage').merge(probabilities, on=['fish', 'cycle', 'phase', 'endpoint'], validate='one_to_one')
            rows.append(random_result)
        fold += 1
        start += pd.Timedelta(days=7)
    if not rows:
        raise ValueError(f'No eligible paired test support in {name}')
    result = pd.concat(rows, ignore_index=True)
    result.to_parquet(OUT / f'Coverage {name}.parquet', index=False)
    pd.DataFrame(audit).to_csv(OUT / f'Fold support {name}.csv', index=False)
    if name == 'primary':
        pd.DataFrame(allocations).to_csv(OUT / 'Training allocations.csv', index=False)
        result.to_csv(OUT / 'Coverage primary.csv', index=False)
        pd.DataFrame(random_distribution).to_csv(OUT / 'Random configuration distribution.csv', index=False)
    print(name, len(result), 'fish', result['fish'].nunique(), 'folds', result['fold'].nunique(), flush=True)
    return name


def bootstrap_paired(z, first, second, draws=4000):
    a = z.loc[z['method'].isin([first, second])].pivot(index=['fish', 'cycle', 'fold'], columns='method', values='coverage').dropna()
    if a.empty:
        return {}
    a['delta'] = a[first] - a[second]
    fish = sorted(a.index.get_level_values('fish').unique())
    folds = sorted(a.index.get_level_values('fold').unique())
    target = a['delta'].groupby('fish').mean().mean()
    totals = a['delta'].groupby(['fish', 'fold']).sum().unstack().reindex(index=fish, columns=folds).fillna(0).to_numpy()
    counts = a['delta'].groupby(['fish', 'fold']).size().unstack().reindex(index=fish, columns=folds).fillna(0).to_numpy()
    rng = np.random.default_rng(SEED)
    samples = np.empty(draws)
    for b in range(draws):
        f = rng.integers(len(fish), size=len(fish))
        t = rng.integers(len(folds), size=len(folds))
        numerator = totals[f][:, t].sum(axis=1)
        denominator = counts[f][:, t].sum(axis=1)
        valid = denominator > 0
        samples[b] = np.mean(numerator[valid] / denominator[valid]) if valid.any() else np.nan
    samples = samples[np.isfinite(samples)]
    return {'first': first, 'second': second, 'difference_pp': 100 * float(target),
            'interval_low_pp': 100 * float(np.quantile(samples, .025)),
            'interval_high_pp': 100 * float(np.quantile(samples, .975)),
            'fish_n': len(fish), 'test_block_n': len(folds), 'fish_cycle_n': len(a),
            'bootstrap_kind': 'Fish and 7-day block product bootstrap; exploratory, single reservoir'}


def summarize():
    results = pd.concat([pd.read_parquet(p) for p in sorted(OUT.glob('Coverage *.parquet'))], ignore_index=True)
    per_fish = results.groupby(['configuration', 'method', 'k', 'phase', 'endpoint', 'fish'], as_index=False).agg(
        coverage=('coverage', 'mean'), cycle_n=('cycle', 'nunique'), test_block_n=('fold', 'nunique'))
    per_fish.to_csv(OUT / 'Per fish coverage.csv', index=False)
    summary = per_fish.groupby(['configuration', 'method', 'k', 'phase', 'endpoint'], as_index=False).agg(
        coverage=('coverage', 'mean'), fish_n=('fish', 'nunique'), minimum_fish_coverage=('coverage', 'min'),
        maximum_fish_coverage=('coverage', 'max'))
    summary.to_csv(OUT / 'Coverage summaries.csv', index=False)
    positions = results.loc[results['endpoint'] == 'Position'].pivot(index=['configuration', 'method', 'k', 'fish', 'cycle', 'fold'], columns='phase', values='coverage').dropna()
    positions['worse_phase_coverage'] = positions[['Day', 'Night']].min(axis=1)
    worst_fish = positions['worse_phase_coverage'].groupby(['configuration', 'method', 'k', 'fish']).mean().reset_index()
    worst_fish.to_csv(OUT / 'Worse phase per fish coverage.csv', index=False)
    worst_fish.groupby(['configuration', 'method', 'k'], as_index=False).agg(coverage=('worse_phase_coverage', 'mean'), fish_n=('fish', 'nunique')).to_csv(OUT / 'Worse phase summaries.csv', index=False)
    primary = results.loc[(results['configuration'] == 'primary') & (results['k'] == 8) & (results['endpoint'] == 'Position')]
    comparisons = []
    for phase in ('Day', 'Night'):
        for first, second in [('Night hotspot', 'Day hotspot'), ('Pooled hotspot', 'Day hotspot'),
                              ('Spatial balance', 'Day hotspot'), ('State balance', 'Day hotspot'),
                              ('State balance', 'Pooled hotspot'), ('State balance', 'Phase quota'),
                              ('State balance', 'Night hotspot')]:
            stats = bootstrap_paired(primary.loc[primary['phase'] == phase], first, second)
            stats['phase'] = phase
            comparisons.append(stats)
    pd.DataFrame(comparisons).to_csv(OUT / 'Primary paired comparisons.csv', index=False)
    primary_summary = summary.loc[(summary['configuration'] == 'primary') & (summary['k'] == 8) & (summary['endpoint'] == 'Position')]
    print(primary_summary.to_string(index=False), flush=True)
    print(pd.DataFrame(comparisons).to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'run', 'summarize', 'all'], default='all')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.stage in ['prepare', 'all']:
        prepare()
    if args.stage in ['run', 'all']:
        settings = [('primary', 100, 75, 5, 12, 'raw'), ('grid50', 50, 75, 5, 12, 'raw'),
                    ('grid200', 200, 75, 5, 12, 'raw'), ('residual25', 100, 25, 5, 12, 'raw'),
                    ('residual50', 100, 50, 5, 12, 'raw'), ('sampling10', 100, 75, 10, 6, 'raw'),
                    ('samples24', 100, 75, 5, 24, 'raw'), ('gam_coordinates', 100, 75, 5, 12, 'gam')]
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            list(executor.map(run_configuration, settings))
    if args.stage in ['summarize', 'all']:
        summarize()
