#  _____ _____ ____   ____
# |  ___|  ___|  _ \ / ___|
# | |_  | |_  | |_) | |
# |  _| |  _| |  _ <| |___
# |_|   |_|   |_| \_\____|
# FFRC
from pathlib import Path
import numpy as np
import pandas as pd
import json

import argparse
parser=argparse.ArgumentParser(description='覆盖损失分解')
parser.add_argument('--inputs',type=Path,required=True)
parser.add_argument('--prior',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
BASE=args.inputs.resolve(); PRE=args.prior.resolve(); OUT=args.out.resolve()
OUT.mkdir(parents=True,exist_ok=True)

def add_parts(d):
    d=d.copy()
    d['scope_loss']=1-d.available
    d['budget_loss']=d.available-d.oracle
    d['rule_gap']=d.oracle-d.coverage
    d['efficiency']=d.coverage/d.oracle.replace(0,np.nan)
    assert d[['scope_loss','budget_loss','rule_gap']].min().min()>-1e-10
    np.testing.assert_allclose(d.coverage+d.scope_loss+d.budget_loss+d.rule_gap,1,atol=1e-10)
    return d

o=pd.read_csv(PRE/'Optical allocation_20261021.csv')
o['coverage']=o.recall
o['available']=(o.targets-o.outside_candidate_targets)/o.targets
o['oracle']=np.minimum(o.budget,o.targets-o.outside_candidate_targets)/o.targets
o=add_parts(o)
o.to_csv(OUT/'Optical decomposition_20261021.csv',index=False)

c=pd.read_csv(PRE/'Monthly chemical allocation_20261021.csv')
exact=c[c.indicator.ne('TP')].copy()
exact['oracle_hits']=np.minimum(exact.budget,exact.targets_certain)
e=exact.groupby(['indicator','rule','budget'],as_index=False)[['hit_lo','oracle_hits','targets_certain']].sum()
e['coverage']=e.hit_lo/e.targets_certain
e['available']=1.
e['oracle']=e.oracle_hits/e.targets_certain
e=add_parts(e)
e.to_csv(OUT/'Chemical decomposition_20261021.csv',index=False)

# TP各项是保守边界，不能把分别的端点相加当作同一实现。
t=c[c.indicator.eq('TP')].copy()
t['oracle_hits_lo']=np.minimum(t.budget,t.targets_certain)
t['oracle_hits_hi']=np.minimum(t.budget,t.targets_possible)
t=t.groupby(['rule','budget'],as_index=False).sum(numeric_only=True)
prev=pd.read_csv(PRE/'Chemical allocation_20261021.csv')
t=t.merge(prev[prev.indicator.eq('TP')][['rule','budget','recall_lo','recall_hi']],on=['rule','budget'],validate='one_to_one')
t['oracle_lo']=t.oracle_hits_lo/t.targets_possible
t['oracle_hi']=np.minimum(1,t.oracle_hits_hi/t.targets_certain)
t.loc[t.budget.eq(10),['oracle_lo','oracle_hi']]=1.
t['scope_loss_lo']=0.;t['scope_loss_hi']=0.
t['budget_loss_lo']=1-t.oracle_hi;t['budget_loss_hi']=1-t.oracle_lo
t['rule_gap_lo']=np.maximum(0,t.oracle_lo-t.recall_hi)
t['rule_gap_hi']=np.minimum(1,t.oracle_hi-t.recall_lo)
t.to_csv(OUT/'TP conservative bounds_20261021.csv',index=False)

images=pd.read_csv(BASE/'reservoir innovation v28/imagery/All budget source coverage.csv')
images['targets']=images.dataset.map({'UAV':20,'underwater':28})
# 原表的source_coverage_fraction指所有来源，不能代替阳性来源召回。
images['coverage']=images.expected_positive_sources/images.targets
images['available']=1.
images['oracle']=np.minimum(images.budget,images.targets)/images.targets
images=add_parts(images)
images.to_csv(OUT/'Image source decomposition_20261021.csv',index=False)
completed=pd.read_csv(BASE/'Review reinforcement v42 1010/Imaging/Completed all budget curves 1010.csv')
completed['coverage']=completed.positive_video_recall
completed['available']=1.
completed['oracle']=np.minimum(completed.budget,completed.reference_positive_videos)/completed.reference_positive_videos
completed=add_parts(completed)
completed.to_csv(OUT/'Completed video decomposition_20261021.csv',index=False)

v=BASE/'reservoir ecology v25/outputs'
d=pd.read_parquet(v/'Primary sampled positions.parquet');d['cycle']=d.cycle.astype(str)
folds=pd.read_csv(v/'Fold support primary.csv')
alloc=pd.read_csv(v/'Training allocations.csv')
future=[];candidates={}
for f in folds.itertuples():
    ids=f.training_fish.split('|')
    tr=d[(d.cycle>=f.train_start)&(d.cycle<f.test_start)&d.fish.isin(ids)]
    te=d[(d.cycle>=f.test_start)&(d.cycle<f.test_end_exclusive)&d.fish.isin(ids)].copy()
    te=te[te.groupby(['fish','cycle']).phase.transform('nunique').eq(2)]
    te['fold']=f.fold;future.append(te)
    candidates[f.fold]=set(tr.cell)
future=pd.concat(future,ignore_index=True)
assert len(future[['fish','cycle']].drop_duplicates())==229
assert future.fish.nunique()==5

def telemetry(data,omit=-1):
    data=data.copy()
    points=data.groupby(['fish','cycle','phase']).cell.transform('size')
    cycles=data[['fish','cycle']].drop_duplicates().groupby('fish').size()
    data['weight']=1/points/data.fish.map(cycles)/data.fish.nunique()/2
    assert abs(data.weight.sum()-1)<1e-12
    rows=[]
    for fold,g in data.groupby('fold'):
        mass=g.groupby('cell').weight.sum()
        inside=mass.reindex(sorted(candidates[fold]),fill_value=0)
        for (method,k),a in alloc[alloc.fold.eq(fold)].groupby(['method','k']):
            cells=set(a.cell)
            assert len(cells)==k and cells<=candidates[fold]
            hit=mass.reindex(sorted(cells),fill_value=0).sum()
            oracle=inside.nlargest(k).sum()
            rows.append(dict(omit=omit,fold=int(fold),method=method,budget=int(k),
                coverage=hit,available=inside.sum(),oracle=oracle,total=g.weight.sum()))
    z=pd.DataFrame(rows)
    agg=z.groupby(['omit','method','budget'],as_index=False)[['coverage','available','oracle','total']].sum()
    assert np.allclose(agg.total,1)
    return add_parts(agg),z

tele,foldwise=telemetry(future)
tele.to_csv(OUT/'Telemetry decomposition_20261021.csv',index=False)
foldwise.to_csv(OUT/'Telemetry fold contributions_20261021.csv',index=False)
loo=[]
for fold in sorted(future.fold.unique()):
    x,_=telemetry(future[future.fold.ne(fold)],int(fold));loo.append(x)
pd.concat(loo).to_csv(OUT/'Telemetry leave-week-out_20261021.csv',index=False)

# 原覆盖独立对照，日夜与鱼权重保持原定义。
old=pd.read_csv(PRE/'Reservoir exclusion days_20261021.csv')
expected=old.groupby(['fish','phase']).original.mean().groupby('phase').mean().mean()
actual=tele[(tele.method=='Pooled hotspot')&(tele.budget==8)].coverage.iloc[0]
assert abs(expected-actual)<1e-10,(expected,actual)
checks=dict(optical_rows=len(o),chemical_exact_rows=len(e),telemetry_rows=len(tele),
            telemetry_paired_cycles=229,baseline_reproduced=float(actual),
            decomposition_identity='PASS',negative_parts='NONE',oracle='exact for additive cell/event objectives',
            TP='conservative envelopes only')
(OUT/'checks_20261021.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
print('OPTICAL k200 target20%')
print(o[(o.budget==200)&(o.target_fraction==.2)][['band','scope','coverage','oracle','scope_loss','budget_loss','rule_gap','efficiency']].to_string(index=False))
print('CHEMICAL k3')
print(e[e.budget==3][['indicator','rule','coverage','oracle','budget_loss','rule_gap','efficiency']].to_string(index=False))
print('TELEMETRY pooled')
print(tele[tele.method.eq('Pooled hotspot')][['budget','coverage','oracle','scope_loss','budget_loss','rule_gap','efficiency']].to_string(index=False))
