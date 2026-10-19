#  _____ _____ ____   ____
# |  ___|  ___|  _ \ / ___|
# | |_  | |_  | |_) | |
# |  _| |  _| |  _ <| |___
# |_|   |_|   |_| \_\____|
# FFRC
"""Exact tie-averaged source coverage audit; scores never use reference labels."""
from pathlib import Path
import csv, json, importlib.util, hashlib
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'Hangbu SAT validation 20261008'
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('evaluation',BASE/'evaluate_predictions_20261019.py')
ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
def read(p):
    with p.open() as f:return list(csv.DictReader(f))
def save(name,rows):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
u=read(BASE/'UAV input/UAV inputs/paired_predictions.csv')
meta={r['original_filename']:r['parent_stem'] for r in read(BASE/'UAV input/UAV inputs/parent_metadata.csv')}
paths,ledger,preds,fish,uncertain,groups,videos=ev.load_inputs(BASE)
datasets={
 'underwater':(np.array(groups),np.array([len(x)>0 for x in fish]),{m:np.array([len(p[m]) for p in preds]) for m in ['static','bsca']}),
 'UAV':(np.array([meta[r['original_filename']] for r in u]),np.array([int(r['gt_count'])>0 for r in u]),{m:np.array([int(r[m+'_pred_count']) for r in u]) for m in ['static','bsca']})}
curves=[];sources=[];summary={}
for dataset,(groups,pos,scores) in datasets.items():
    n=len(pos); ng=len(np.unique(groups)); data={}
    for m,score in scores.items():
        for g in np.unique(groups):
            mask=groups==g
            sources.append(dict(dataset=dataset,method=m,source=str(g),eligible_units=int(mask.sum()),reference_positive_units=int(pos[mask].sum()),predicted_alert_units=int((score[mask]>0).sum()),maximum_predicted_count=int(score[mask].max())))
        data[m+'_global']=ev.global_expected(score,pos,groups,np.arange(1,n+1))
        data[m+'_grouped_first_round']=ev.grouped_expected(score,pos,groups,np.arange(1,ng+1))
    data['uniform_frame_random']=ev.global_expected(np.zeros(n),pos,groups,np.arange(1,n+1))
    data['random_one_frame_per_source']=ev.grouped_expected(np.zeros(n),pos,groups,np.arange(1,ng+1),True)
    ds={}
    for strategy,curve in data.items():
        if strategy.endswith('global'):
            score=scores[strategy.split('_')[0]]
            allsource=ev.global_expected(score,np.ones(n,dtype=bool),groups,np.arange(1,n+1))[:,1]
        elif strategy=='uniform_frame_random':
            allsource=ev.global_expected(np.zeros(n),np.ones(n,dtype=bool),groups,np.arange(1,n+1))[:,1]
        else:allsource=np.arange(1,ng+1,dtype=float)
        for k,((pf,ps),coverage) in enumerate(zip(curve,allsource),1):
            row=dict(dataset=dataset,strategy=strategy,budget=k,expected_positive_units=float(pf),expected_positive_sources=float(ps),expected_all_sources=float(coverage),expected_repeat_units=float(k-coverage),source_coverage_fraction=float(coverage/ng))
            curves.append(row)
            if k==ng:ds[strategy]=row
    for m in scores:
        glob=data[m+'_global'];grp=data[m+'_grouped_first_round']
        ds[m+'_first_round_joint_nondecrease_budgets']=[k for k in range(1,ng+1) if grp[k-1,0]>=glob[k-1,0]-1e-9 and grp[k-1,1]>=glob[k-1,1]-1e-9]
        ds[m+'_first_round_positive_unit_loss_budgets']=[k for k in range(1,ng+1) if grp[k-1,0]<glob[k-1,0]-1e-9]
        ds[m+'_sources_without_any_predicted_alert']=int(sum(scores[m][groups==g].max()==0 for g in np.unique(groups)))
    summary[dataset]=ds
save('All budget source coverage.csv',curves)
save('Source alert census.csv',sources)
summary['provenance']={'score_rule':'Frozen detector counts, unchanged thresholds; uniform exact ties. Reference labels evaluate success only.', 'interpretation':'Source coverage denotes acquisition provenance, not independent ecological sites. Repeats count reviewed units beyond the first from a source, not duplicate fish.', 'input_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths+[BASE/'UAV input/UAV inputs/paired_predictions.csv',BASE/'UAV input/UAV inputs/parent_metadata.csv']}}
(OUT/'Summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps({d:{k:v for k,v in x.items() if isinstance(v,dict) or 'without' in k or 'joint' in k} for d,x in summary.items() if d!='provenance'},indent=2))
