#  _____ _____ ____   ____
# |  ___|  ___|  _ \ / ___|
# | |_  | |_  | |_) | |
# |  _| |  _| |  _ <| |___
# |_|   |_|   |_| \_\____|
# FFRC
from pathlib import Path
import importlib.util,hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent;V28=ROOT.parents[1];SRC=V28/'public assets/NewmanSound_year1.csv';OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True)
spec=importlib.util.spec_from_file_location('objective',V28/'coverage_objectives_20261019.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
d=pd.read_csv(SRC);d['time']=pd.to_datetime(d.Hourbin,format='%d/%m/%Y %H:%M');d['fish']=d.Tag.astype(str);d['cycle']=d.time.dt.strftime('%Y-%m-%d');d['phase']='All';d['hour']=d.time.dt.floor('h');d=d[d.Transplant.eq('Control')].copy();assert d.fish.nunique()==21
assert not d.duplicated(['fish','hour']).any();assert d[['POINT_X','POINT_Y']].notna().all().all()
d['cellx']=np.floor(d.POINT_X/100).astype(int);d['celly']=np.floor(d.POINT_Y/100).astype(int);d['cell']=d.cellx.astype(str)+':'+d.celly.astype(str)
counts=d.groupby(['fish','cycle']).hour.nunique().rename('hour_n');counts.to_csv(OUT/'Control daily support.csv');d=d.join(counts,on=['fish','cycle']);eligible=d[d.hour_n>=12].copy()
rows=[];audits=[];alloc=[];solvers=[];start=pd.Timestamp('2010-10-08');last=d.time.max().normalize();fold=0
while start+pd.Timedelta(days=20)<=last:
 cutoff=start+pd.Timedelta(days=14);end=cutoff+pd.Timedelta(days=7)
 train=eligible[(eligible.cycle>=str(start.date()))&(eligible.cycle<str(cutoff.date()))];n=train.groupby('fish').cycle.nunique();ids=n.index[n>=3].tolist();train=train[train.fish.isin(ids)];test_all=eligible[(eligible.cycle>=str(cutoff.date()))&(eligible.cycle<str(end.date()))];test=test_all[test_all.fish.isin(ids)]
 cells=sorted(train.cell.unique());audit=dict(fold=fold,train_start=str(start.date()),test_start=str(cutoff.date()),test_end_exclusive=str(end.date()),training_ids='|'.join(sorted(ids)),testing_ids='|'.join(sorted(test.fish.unique())),candidate_cells=len(cells),all_future_eligible_fish_days=len(test_all[['fish','cycle']].drop_duplicates()),paired_eligible_fish_days=len(test[['fish','cycle']].drop_duplicates()),status='eligible');audits.append(audit)
 if train.empty or test.empty or len(cells)<8:
  audit['status']='no_evaluable_support';start+=pd.Timedelta(days=7);fold+=1;continue
 probability,labels=m.probability_matrix(train,ids,cells);methods,solver=m.selections(probability,labels,8);solver.update(fold=fold,grid_m=100,training_ids='|'.join(sorted(ids)));solvers.append(solver)
 for method,indices in methods.items():
  selected=[cells[i] for i in indices]
  for cell in selected:alloc.append(dict(fold=fold,method=method,k=8,cell=cell))
  z=test[['fish','cycle','phase']].copy();z['coverage']=test.cell.isin(selected).to_numpy();z['outside_candidate']=~test.cell.isin(cells).to_numpy();z=z.groupby(['fish','cycle','phase'],as_index=False).agg(coverage=('coverage','mean'),outside_candidate=('outside_candidate','mean'),hour_n=('coverage','size'));z['fold']=fold;z['method']=method;z['k']=8;rows.append(z)
 print(json.dumps(dict(fold=fold,fish=len(test.fish.unique()),days=audit['paired_eligible_fish_days'],stage1=solver['stage1_status'],stage2=solver['stage2_status'])),flush=True)
 start+=pd.Timedelta(days=7);fold+=1
r=pd.concat(rows,ignore_index=True);r.to_csv(OUT/'Fish day coverage.csv',index=False);pd.DataFrame(audits).to_csv(OUT/'Fold support.csv',index=False);pd.DataFrame(alloc).to_csv(OUT/'Allocations.csv',index=False);pd.DataFrame(solvers).to_csv(OUT/'Solver audits.csv',index=False)
summaries=[];perfish=[]
for scope,z in [('first_window',r[r.fold==0]),('all_rolling',r)]:
 z=z.copy();z['zero']=z.coverage.eq(0);z['below10']=z.coverage.lt(.1);f=z.groupby(['method','k','fish'],as_index=False).agg(coverage=('coverage','mean'),zero_day_fraction=('zero','mean'),below10_day_fraction=('below10','mean'),outside_candidate=('outside_candidate','mean'),days=('cycle','size'),weeks=('fold','nunique'));f['scope']=scope;perfish.append(f)
 a=f.groupby(['method','k'],as_index=False).agg(mean_coverage=('coverage','mean'),minimum_fish_coverage=('coverage','min'),zero_day_fraction=('zero_day_fraction','mean'),below10_day_fraction=('below10_day_fraction','mean'),outside_candidate=('outside_candidate','mean'),fish_n=('fish','nunique'),fish_days=('days','sum'));a['scope']=scope;summaries.append(a)
pd.concat(perfish,ignore_index=True).to_csv(OUT/'Per fish coverage.csv',index=False);summary=pd.concat(summaries,ignore_index=True);summary.to_csv(OUT/'Summary.csv',index=False)
r.groupby(['method','fish','fold'],as_index=False).agg(coverage=('coverage','mean'),days=('cycle','size'),outside_candidate=('outside_candidate','mean')).to_csv(OUT/'Per fish week.csv',index=False)
data=SRC.read_bytes();assert hashlib.md5(data).hexdigest()=='399b41918fe7892142fc6edb505d8b63';(OUT/'Source and execution audit.json').write_text(json.dumps(dict(source=str(SRC.relative_to(V28)),bytes=len(data),md5=hashlib.md5(data).hexdigest(),sha256=hashlib.sha256(data).hexdigest(),control_identity_n=21,control_source_rows=len(d),qualified_rows=len(eligible),qualified_fish_days=len(eligible[['fish','cycle']].drop_duplicates()),calendar_time_basis='source reported time, timezone unspecified',total_full_folds=len(audits),evaluable_folds=len(solvers),unevaluated_tail_start=str((start+pd.Timedelta(days=14)).date()),last_source_day=str(last.date()),methods='Pooled hotspot / Individual window maximin',grid_m=100,k=8,both_stage_optimal_folds=sum(x['both_stages_optimal_to_solver_tolerance'] for x in solvers)),indent=2)+'\n');print(summary.to_string(index=False),flush=True)
