#  _____ _____ ____   ____
# |  ___|  ___|  _ \ / ___|
# | |_  | |_  | |_) | |
# |  _| |  _| |  _ <| |___
# |_|   |_|   |_| \_\____|
# FFRC
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd

import argparse
parser=argparse.ArgumentParser(description='遥感和水质名额分配')
parser.add_argument('--inputs',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
BASE=args.inputs.resolve(); OUT=args.out.resolve()
OUT.mkdir(parents=True,exist_ok=True)

def weights(score, k):
    score = np.asarray(score)
    cutoff = np.sort(score)[-k]
    p = (score > cutoff).astype(float)
    tied = score == cutoff
    p[tied] = (k-p.sum()) / tied.sum()
    assert abs(p.sum()-k) < 1e-8
    return p

s = pd.read_csv(BASE/'daxi/full_core_matched_reflectance.csv')
rows = []
for band, g in s.groupby('band'):
    x = g.reflectance_20250823.to_numpy()
    y = g.reflectance_20250825.to_numpy()
    candidates = g.unique_camera_GPS_pixel.astype(str).str.lower().isin(['true','1']).to_numpy()
    for frac in [.1,.2,.3]:
        target = y >= np.quantile(y,1-frac)
        for scope, mask in [('full',np.ones(len(g),bool)),('camera_centres',candidates)]:
            for k in [25,50,100,200,400,800]:
                if mask.sum() < k: continue
                hit = float(weights(x[mask],k) @ target[mask])
                rows.append(dict(band=band,target_fraction=frac,scope=scope,budget=k,
                    candidates=int(mask.sum()),targets=int(target.sum()),expected_hits=hit,
                    recall=hit/target.sum(),random_recall=k/mask.sum()*target[mask].sum()/target.sum(),
                    outside_candidate_targets=int(target[~mask].sum()),
                    unselected_inside_targets=float(target[mask].sum()-hit)))
pd.DataFrame(rows).to_csv(OUT/'Optical allocation_20261021.csv',index=False)

d = pd.read_csv(BASE/'field/Triadic interval states.csv')
d = d[(d.year==2024)&(d.cohort=='balanced_7months')&(d['mode']=='censor_interval')]
rows=[]
for indicator,g in d.groupby('indicator'):
    assert len(g)==70
    for month in range(7,13):
        a=g[g.month==month-1].set_index('site_original').sort_index()
        b=g[g.month==month].set_index('site_original').reindex(a.index)
        assert len(a)==len(b)==10 and not b.WS_lo.isna().any()
        certain=(b.EP_lo>b.WS_hi).to_numpy()
        possible=(b.EP_hi>b.WS_lo).to_numpy()
        for rule,ref in [('source','WS'),('pond','CP'),('endpoint',None),('random',None)]:
            lo=(a.EP_lo-a[ref+'_hi'] if ref else a.EP_lo).to_numpy()
            hi=(a.EP_hi-a[ref+'_lo'] if ref else a.EP_hi).to_numpy()
            for k in range(1,11):
                if rule=='random' or (np.array_equal(lo,hi) and np.array_equal(certain,possible)):
                    p=np.full(10,k/10) if rule=='random' else weights(lo,k)
                    hit_lo=float(p@certain); hit_hi=float(p@possible)
                    outside_lo=float((1-p)@certain); outside_hi=float((1-p)@possible)
                else:
                    bounds=[]
                    for comb in combinations(range(10),k):
                        sel=np.zeros(10,bool); sel[list(comb)]=True
                        if k<10 and min(hi[sel])<max(lo[~sel]): continue
                        bounds.append((certain[sel].sum(),possible[sel].sum(),certain[~sel].sum(),possible[~sel].sum()))
                    v=np.array(bounds); assert len(v)>0
                    hit_lo=float(v[:,0].min()); hit_hi=float(v[:,1].max())
                    outside_lo=float(v[:,2].min()); outside_hi=float(v[:,3].max())
                rows.append(dict(indicator=indicator,month=month,rule=rule,budget=k,
                    hit_lo=hit_lo,hit_hi=hit_hi,outside_lo=outside_lo,outside_hi=outside_hi,
                    targets_certain=int(certain.sum()),targets_possible=int(possible.sum())))
r=pd.DataFrame(rows)
r.to_csv(OUT/'Monthly chemical allocation_20261021.csv',index=False)
z=r.groupby(['indicator','rule','budget'],as_index=False).sum(numeric_only=True)
z['recall_lo']=z.hit_lo/(z.hit_lo+z.outside_hi)
z['recall_hi']=z.hit_hi/(z.hit_hi+z.outside_lo)
# 随机名额对任意非空目标集合的期望召回恒为 k/n。
random_rows=z.rule=='random'
z.loc[random_rows,'recall_lo']=z.loc[random_rows,'budget']/10
z.loc[random_rows,'recall_hi']=z.loc[random_rows,'budget']/10
z.to_csv(OUT/'Chemical allocation_20261021.csv',index=False)
print('Chemical examples, budget 3:')
print(z[z.budget==3][['indicator','rule','targets_certain','targets_possible','recall_lo','recall_hi']].to_string(index=False))
print('Optical examples, budget 200, target 20%:')
o=pd.read_csv(OUT/'Optical allocation_20261021.csv')
print(o[(o.budget==200)&(o.target_fraction==.2)][['band','scope','candidates','targets','recall','random_recall','outside_candidate_targets']].to_string(index=False))
