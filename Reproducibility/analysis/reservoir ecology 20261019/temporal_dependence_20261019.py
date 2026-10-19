#  _____ _____ ____   ____
# |  ___|  ___|  _ \ / ___|
# | |_  | |_  | |_) | |
# |  _| |  _| |  _ <| |___
# |_|   |_|   |_| \_\____|
# FFRC
from pathlib import Path
import json

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'outputs'
z = pd.read_parquet(OUT / 'Coverage primary.parquet')
z = z.loc[(z['k'] == 8) & (z['endpoint'] == 'Position')]
rows = []
for phase in ['Day', 'Night']:
    paired = z.loc[(z['phase'] == phase) & z['method'].isin(['Pooled hotspot', 'Day hotspot'])].pivot(
        index=['fish', 'cycle', 'fold'], columns='method', values='coverage').dropna()
    paired['delta'] = paired['Pooled hotspot'] - paired['Day hotspot']
    fish = sorted(paired.index.get_level_values('fish').unique())
    folds = sorted(paired.index.get_level_values('fold').unique())
    total = paired['delta'].groupby(['fish', 'fold']).sum().unstack().reindex(index=fish, columns=folds).fillna(0).to_numpy()
    count = paired['delta'].groupby(['fish', 'fold']).size().unstack().reindex(index=fish, columns=folds).fillna(0).to_numpy()
    observed = paired['delta'].groupby('fish').mean().mean()
    # 相邻3周作为一个重采样块，额外检查共享历史与串行相关。
    rng = np.random.default_rng(20261008)
    bootstrap = []
    for _ in range(4000):
        sampled_fish = rng.integers(len(fish), size=len(fish))
        starts = rng.integers(len(folds), size=int(np.ceil(len(folds) / 3)))
        sampled_folds = np.concatenate([(s + np.arange(3)) % len(folds) for s in starts])[:len(folds)]
        numerator = total[sampled_fish][:, sampled_folds].sum(axis=1)
        denominator = count[sampled_fish][:, sampled_folds].sum(axis=1)
        valid = denominator > 0
        if valid.any():
            bootstrap.append(np.mean(numerator[valid] / denominator[valid]))
    rows.append({'phase': phase, 'comparison': 'Pooled minus Day hotspot', 'kind': 'Fish and circular three-week block bootstrap',
                 'difference_pp': 100 * observed, 'low_pp': 100 * np.quantile(bootstrap, .025),
                 'high_pp': 100 * np.quantile(bootstrap, .975), 'fish_n': len(fish), 'fish_cycle_n': len(paired)})
    # 三种相位的21日间隔验证；每组内训练与测试支持都不交叠。
    for offset in range(3):
        sub = paired.loc[paired.index.get_level_values('fold') % 3 == offset]
        rows.append({'phase': phase, 'comparison': 'Pooled minus Day hotspot',
                     'kind': f'Nonoverlapping 21-day analysis blocks, offset {offset}',
                     'difference_pp': 100 * sub['delta'].groupby('fish').mean().mean(),
                     'low_pp': None, 'high_pp': None,
                     'fish_n': sub.index.get_level_values('fish').nunique(), 'fish_cycle_n': len(sub)})
pd.DataFrame(rows).to_csv(OUT / 'Temporal dependence sensitivity.csv', index=False)
print(pd.DataFrame(rows).to_string(index=False))
