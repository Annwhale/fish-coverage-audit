#  _____ _____ ____   ____
# |  ___|  ___|  _ \ / ___|
# | |_  | |_  | |_) | |
# |  _| |  _| |  _ <| |___
# |_|   |_|   |_| \_\____|
# FFRC
from pathlib import Path
import importlib.util, json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
V25=ROOT.parents[1]/'reservoir ecology v25'
spec=importlib.util.spec_from_file_location('v25',V25/'analyse_rimov.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
OUT=ROOT/'outputs';OUT.mkdir(exist_ok=True)
METHODS=['Day hotspot','Pooled hotspot','State balance'];KS=[1,2,4,8,16,32]
s=pd.read_parquet(V25/'outputs/Primary sampled positions.parquet');s['cycle']=s.cycle.astype(str)
folds=pd.read_csv(V25/'outputs/Fold support primary.csv');alloc=pd.read_csv(V25/'outputs/Training allocations.csv');original=sorted(set('|'.join(folds.testing_fish).split('|')))
records=[];audits=[];selections=[]
def paired(z):
 p=z.groupby(['fish','cycle']).phase.nunique().eq(2);return z.join(p.rename('_paired'),on=['fish','cycle']).query('_paired').drop(columns='_paired')
def fit(z,exclude=None):
 # 仅过去合格动物，留出鱼不参与候选或坐标中心。
 if exclude is not None:z=z[z.fish!=exclude]
 n=z[['fish','cycle','phase']].drop_duplicates().groupby(['fish','phase']).size().unstack(fill_value=0)
 if not {'Day','Night'}.issubset(n.columns):return None
 ids=n.index[(n.Day>=3)&(n.Night>=3)].tolist();z=z[z.fish.isin(ids)]
 if z.empty:return None
 cells=sorted(z.cell.unique());cent=z[['cell','cellx','celly']].drop_duplicates('cell').set_index('cell').reindex(cells)
 orders=m.configuration_orders(m.cell_probability(z,ids,'Day',cells),m.cell_probability(z,ids,'Night',cells),(cent[['cellx','celly']].to_numpy()+.5)*100)
 return cells,orders,ids

def evaluate(z,cells,method,k,design,fold,heldout='',cohort='all',candidate=None):
 if z.empty:return
 v=z[['fish','cycle','phase']].copy();v['coverage']=z.cell.isin(cells).to_numpy();v['outside_candidate']=~z.cell.isin(candidate if candidate is not None else cells).to_numpy()
 a=v.groupby(['fish','cycle','phase'],as_index=False).agg(coverage=('coverage','mean'),sample_n=('coverage','size'),outside_candidate=('outside_candidate','mean'))
 for key,value in dict(design=design,fold=fold,heldout=heldout,cohort=cohort,method=method,k=k).items():a[key]=value
 records.append(a)

# A完全复用原始选择，并从原季节位置核全部动物。
for f in folds.itertuples():
 test=paired(s[(s.cycle>=f.test_start)&(s.cycle<f.test_end_exclusive)]);train_ids=f.training_fish.split('|')
 audits.append(dict(design='A',fold=f.fold,test_start=f.test_start,test_end=f.test_end_exclusive,pair_cycles=len(test[['fish','cycle']].drop_duplicates()),new_pair_cycles=len(test[~test.fish.isin(train_ids)][['fish','cycle']].drop_duplicates())))
 # 候选还原仅用原训练合格鱼。
 candidate=s[(s.cycle>=f.train_start)&(s.cycle<f.test_start)&s.fish.isin(train_ids)].cell.unique()
 for method in METHODS:
  for k in KS:
   sel=alloc[(alloc.fold==f.fold)&(alloc.method==method)&(alloc.k==k)].cell.tolist()
   if len(sel)!=k:continue
   for cohort,z in [('all',test),('original_eligible',test[test.fish.isin(train_ids)]),('new',test[~test.fish.isin(train_ids)])]:evaluate(z,sel,method,k,'A',f.fold,cohort=cohort,candidate=candidate)

# B过去拟合与留出动物严格分开。
for f in folds.itertuples():
 train=s[(s.cycle>=f.train_start)&(s.cycle<f.test_start)];future=paired(s[(s.cycle>=f.test_start)&(s.cycle<f.test_end_exclusive)])
 for fish in original:
  fitres=fit(train,fish);test=future[future.fish==fish]
  audits.append(dict(design='B',fold=f.fold,heldout=fish,pair_cycles=len(test[['fish','cycle']].drop_duplicates()),candidate_cells=0 if fitres is None else len(fitres[0]),train_start=f.train_start,test_start=f.test_start,test_end=f.test_end_exclusive,training_ids='' if fitres is None else '|'.join(fitres[2])))
  if fitres is None:continue
  cells,orders,ids=fitres
  for method in METHODS:
   for k in KS:
    if len(cells)<k:continue
    selected=[cells[i] for i in orders[method][:k]]
    for cell in selected:selections.append(dict(design='B',fold=f.fold,heldout=fish,method=method,k=k,cell=cell))
    evaluate(test,selected,method,k,'B',f.fold,heldout=fish,candidate=cells)
    if fish in f.training_fish.split('|'):
     evaluate(test,selected,method,k,'B',f.fold,heldout=fish,cohort='original_eligible',candidate=cells)

# C读取发布后全源；不修改v25缓存，源标记不再解释为未处理声学真值。
raw=pd.read_csv(V25/'alternative assets/zenodo_rimov_19885546/raw_positions.csv')
raw['time']=pd.to_datetime(raw.up_timestamp_utc,utc=True,format='mixed');local=raw.time.dt.tz_convert('Europe/Prague');raw['cycle']=(local-pd.Timedelta(hours=4)).dt.date.astype(str);h=local.dt.hour
raw['phase']=np.select([(h>=8)&(h<16),(h>=22)|(h<4)],['Day','Night'],default='Transition');raw['fish']=raw.fi_fishid.astype(str)
raw=raw[raw.up_validpos.eq(True)&raw.up_fishvalid.eq(True)&raw.fi_fishid.notna()&raw[['up_easting','up_northing','time']].notna().all(axis=1)&raw.phase.isin(['Day','Night'])&raw.cycle.ge('2017-09-11')&raw.cycle.lt('2017-11-21')].copy()
raw=raw.sort_values(['fish','time','up_gamres']).drop_duplicates(['fish','time']);c,counts=m.sample_rows(raw,75,5,12,'raw');c['cellx']=np.floor(c.x/100).astype(int);c['celly']=np.floor(c.y/100).astype(int);c['cell']=c.cellx.astype(str)+':'+c.celly.astype(str)
counts.to_csv(OUT/'C sampling counts.csv');fitres=fit(c[(c.cycle>='2017-09-11')&(c.cycle<'2017-09-25')]);cells,orders,ids=fitres
future=c[(c.cycle>='2017-10-01')&(c.cycle<'2017-11-21')];pc=paired(future)
audits.append(dict(design='C',fold=-1,train_start='2017-09-11',test_start='2017-10-01',test_end='2017-11-21',freeze_cutoff='2017-09-25',training_ids='|'.join(ids),candidate_cells=len(cells),pair_cycles=len(pc[['fish','cycle']].drop_duplicates()),test_ids='|'.join(sorted(future.fish.unique()))))
for method in METHODS:
 for k in KS:
  if len(cells)<k:continue
  selected=[cells[i] for i in orders[method][:k]]
  for cell in selected:selections.append(dict(design='C',fold=-1,heldout='',method=method,k=k,cell=cell))
  for support,z in [('paired',pc),('window_available',future)]:
   for cohort,t in [('all',z),('original5',z[z.fish.isin(original)])]:evaluate(t,selected,method,k,'C_'+support,-1,cohort=cohort,candidate=cells)
r=pd.concat(records,ignore_index=True);r.to_csv(OUT/'Fish cycle coverage.csv',index=False);pd.DataFrame(audits).to_csv(OUT/'Support audits.csv',index=False);pd.DataFrame(selections).to_csv(OUT/'Validation allocations.csv',index=False)
keys=['design','cohort','method','k','phase'];fish=r.groupby(keys+['fish'],as_index=False).agg(coverage=('coverage','mean'),outside_candidate=('outside_candidate','mean'),cycles=('cycle','size'));fish.to_csv(OUT/'Per fish coverage.csv',index=False)
summary=fish.groupby(keys,as_index=False).agg(mean_coverage=('coverage','mean'),minimum_fish_coverage=('coverage','min'),mean_outside_candidate=('outside_candidate','mean'),fish_n=('fish','nunique'),fish_cycles=('cycles','sum'));summary.to_csv(OUT/'Coverage summary.csv',index=False)
weekly=r.groupby(keys+['fish','fold'],as_index=False).agg(coverage=('coverage','mean'),outside_candidate=('outside_candidate','mean'),cycles=('cycle','size'));weekly.to_csv(OUT/'Per fish week coverage.csv',index=False)
(OUT/'Audit.json').write_text(json.dumps(dict(original_five=original,last_original_fold=folds.iloc[-1].to_dict(),A_added_pair_cycles=sum(a.get('new_pair_cycles',0) for a in audits),B_fish_week_supports=sum(a.get('pair_cycles',0)>0 for a in audits if a['design']=='B'),C_publisher_filtered=True,columns=r.columns.tolist()),indent=2)+'\n')
print(summary.query('k==8').to_string(index=False))
