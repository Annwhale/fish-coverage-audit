#  _____ _____ ____   ____
# |  ___|  ___|  _ \ / ___|
# | |_  | |_  | |_) | |
# |  _| |  _| |  _ <| |___
# |_|   |_|   |_| \_\____|
# FFRC
"""FFRC：历史排除与分块扰动。"""
from pathlib import Path
import json, hashlib, sys
import numpy as np
import pandas as pd

import argparse
parser=argparse.ArgumentParser(description='历史排除与分块扰动')
parser.add_argument('--inputs',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
BASE=args.inputs.resolve(); OUT=args.out.resolve(); ROOT=BASE.parent
OUT.mkdir(parents=True,exist_ok=True)
V25 = BASE / 'reservoir ecology v25'
V28 = BASE / 'reservoir innovation v28'
RNG = np.random.default_rng(20261021)
REPS = 300
METRICS = ['original','coverage','shared_loss','outside_loss','gain','net_loss','shared_opportunity','outside_opportunity']

def save(df, name):
    df.to_csv(OUT / (name + '_20261021.csv'), index=False, float_format='%.12g')

def rank(score, cells, k=8):
    return [cells[i] for i in np.lexsort((np.arange(len(cells)), -score))[:k]]

def score(d, cells):
    if set(d.phase)=={'All'}:
        probabilities=[]
        for fish in sorted(d.fish.unique()):
            z=d[d.fish.eq(fish)]
            counts=z.groupby('cycle').size()
            weights=1/z.cycle.map(counts)/len(counts)
            probabilities.append(weights.groupby(z.cell).sum().reindex(cells,fill_value=0).to_numpy())
        return np.asarray(probabilities).mean(axis=0)
    probabilities=[]
    for phase in ['Day','Night']:
        z=d[d.phase.eq(phase)]
        counts=z.groupby(['fish','cycle']).fish.transform('size')
        cycles=z[['fish','cycle']].drop_duplicates().groupby('fish').size()
        weights=1/counts/z.fish.map(cycles)/len(cycles)
        probabilities.append(weights.groupby(z.cell).sum().reindex(cells,fill_value=0).to_numpy())
    return np.mean(probabilities,axis=0)

def evaluate(t, a, b, support):
    z=t[['fish','cycle','phase']].copy()
    ca=t.cell.isin(a).to_numpy(); cb=t.cell.isin(b).to_numpy(); inside=t.cell.isin(support).to_numpy()
    loss=ca&~cb
    vals=[ca,cb,loss&inside,loss&~inside,~ca&cb,ca.astype(float)-cb,ca&inside,ca&~inside]
    for key,val in zip(METRICS,vals): z[key]=np.asarray(val,float)
    z=z.groupby(['fish','cycle','phase'],as_index=False)[METRICS].mean()
    np.testing.assert_allclose(z.net_loss,z.shared_loss+z.outside_loss-z.gain,atol=1e-14)
    return z

def aggregate(d, keys):
    fish=d.groupby(keys+['fish'],as_index=False)[METRICS].mean()
    return fish.groupby(keys,as_index=False)[METRICS].mean()

def marine():
    ext=V28/'validation/external/outputs'
    d=pd.read_csv(V28/'public assets/NewmanSound_year1.csv')
    d=d[d.Transplant.eq('Control')].copy()
    d['time']=pd.to_datetime(d.Hourbin,format='%d/%m/%Y %H:%M')
    d['fish']=d.Tag.astype(str); d['phase']='All'; d['cycle']=d.time.dt.strftime('%Y-%m-%d')
    d['cell']=np.floor(d.POINT_X/100).astype(int).astype(str)+':'+np.floor(d.POINT_Y/100).astype(int).astype(str)
    d=d[d.groupby(['fish','cycle']).Hourbin.transform('nunique')>=12].copy()
    folds=pd.read_csv(ext/'Fold support.csv'); alloc=pd.read_csv(ext/'Allocations.csv')
    rows=[]; checks=[]
    for f in folds.itertuples():
        ids=f.training_ids.split('|')
        tr=d[(d.cycle>=f.train_start)&(d.cycle<f.test_start)&d.fish.isin(ids)]
        cells=sorted(tr.cell.unique())
        if len(cells)<8:
            checks.append(dict(fold=f.fold,fish='',status='original_candidates_lt8',candidates=len(cells))); continue
        a=rank(score(tr,cells),cells)
        saved=alloc[(alloc.fold==f.fold)&alloc.method.eq('Pooled hotspot')&(alloc.k==8)].cell.tolist()
        assert a==saved, ('marine allocation',f.fold)
        te=d[(d.cycle>=f.test_start)&(d.cycle<f.test_end_exclusive)&d.fish.isin(ids)]
        for fish,t in te.groupby('fish'):
            other=tr[tr.fish.ne(fish)]; support=sorted(other.cell.unique())
            if len(support)<8:
                checks.append(dict(fold=f.fold,fish=fish,status='excluded_candidates_lt8',candidates=len(cells),other_candidates=len(support))); continue
            b=rank(score(other,support),support)
            z=evaluate(t,a,b,support); z['fold']=f.fold; rows.append(z)
            checks.append(dict(fold=f.fold,fish=fish,status='ok',candidates=len(cells),other_candidates=len(support),changed=len(set(a)-set(b)),jaccard=len(set(a)&set(b))/len(set(a)|set(b)),fish_days=len(z)))
    result=pd.concat(rows,ignore_index=True); check=pd.DataFrame(checks)
    save(result,'Marine paired days'); save(check,'Marine support and configurations')
    summary=aggregate(result,['phase']); weekly=aggregate(result,['fold','phase'])
    save(summary,'Marine summary'); save(weekly,'Marine weekly summary')
    first=aggregate(result[result.fold.eq(0)],['phase']).iloc[0]
    assert abs(first.original-0.9454365079365079)<1e-12 and abs(first.net_loss)<1e-14
    print('marine',summary.to_dict('records'),flush=True)
    return result,check

def matrices(tr,cells):
    # 每尾鱼以周期为块，昼夜联合抽样。
    groups={}
    for fish,z in tr.groupby('fish'):
        days=sorted(z.cycle.unique()); m=np.zeros((len(days),2,len(cells)))
        for phase_i,phase in enumerate(['Day','Night']):
            q=z[z.phase.eq(phase)].groupby(['cycle','cell']).size().unstack(fill_value=0).reindex(index=days,columns=cells,fill_value=0)
            ar=q.to_numpy(float); total=ar.sum(axis=1)
            m[:,phase_i,:]=np.divide(ar,total[:,None],out=np.zeros_like(ar),where=total[:,None]>0)
        groups[fish]=m
    return groups

def from_matrices(groups,indices):
    probs=[]
    for fish,m in groups.items():
        z=m[indices[fish]]
        for ph in range(2):
            valid=z[:,ph,:].sum(axis=1)>0
            if valid.sum()==0:return None
            probs.append(z[valid,ph,:].mean(axis=0))
    return np.mean(probs,axis=0)

def compact_future(te,cells):
    # 将未来记录压为周期频率，避免重复展开定位点。
    records=[]; arr=[]
    for key,z in te.groupby(['fish','cycle','phase']):
        records.append(key)
        arr.append(z.cell.value_counts().reindex(cells,fill_value=0).to_numpy(float)/len(z))
    return pd.DataFrame(records,columns=['fish','cycle','phase']),np.asarray(arr)

def reservoir():
    d=pd.read_parquet(V25/'outputs/Primary sampled positions.parquet');d['cycle']=d.cycle.astype(str)
    folds=pd.read_csv(V25/'outputs/Fold support primary.csv')
    frozen=pd.read_csv(V25/'outputs/Training allocations.csv')
    excluded=pd.read_csv(V28/'validation/outputs/Validation allocations.csv')
    base=[];boot=[];controls=[];support=[]
    for f in folds.itertuples():
        ids=f.training_fish.split('|')
        tr=d[(d.cycle>=f.train_start)&(d.cycle<f.test_start)&d.fish.isin(ids)].copy()
        te=d[(d.cycle>=f.test_start)&(d.cycle<f.test_end_exclusive)&d.fish.isin(ids)].copy()
        te=te[te.groupby(['fish','cycle']).phase.transform('nunique').eq(2)]
        cells=sorted(tr.cell.unique()); p=score(tr,cells);a=rank(p,cells)
        sa=frozen[(frozen.fold==f.fold)&frozen.method.eq('Pooled hotspot')&frozen.k.eq(8)].cell.tolist()
        assert a==sa,('reservoir allocation',f.fold)
        groups=matrices(tr,cells); indices={fish:np.arange(len(m)) for fish,m in groups.items()}
        np.testing.assert_allclose(from_matrices(groups,indices),p,atol=1e-14)
        future_cells=sorted(set(cells)|set(te.cell)); keys,ff=compact_future(te,future_cells)
        amap=np.isin(future_cells,a); orig=ff[:,amap].sum(axis=1)
        rejected=0
        for rep in range(REPS):
            while True:
                draws={}
                for fish,m in groups.items():
                    n=len(m); starts=RNG.integers(0,n,size=(n+2)//3)
                    draws[fish]=((starts[:,None]+np.arange(3))%n).ravel()[:n]
                bp=from_matrices(groups,draws)
                if bp is not None and (bp>0).sum()>=8:break
                rejected+=1
                assert rejected<100000
            b=rank(bp,cells);cov=ff[:,np.isin(future_cells,b)].sum(axis=1)
            z=keys.copy();z['original']=orig;z['coverage']=cov;z['net_loss']=orig-cov;z['fold']=f.fold;z['rep']=rep
            z['changed']=len(set(a)-set(b));boot.append(z)
        for fish,t in te.groupby('fish'):
            other=tr[tr.fish.ne(fish)];othercells=sorted(other.cell.unique());b=rank(score(other,othercells),othercells)
            sb=excluded[excluded.design.eq('B')&(excluded.fold==f.fold)&excluded.heldout.eq(fish)&excluded.method.eq('Pooled hotspot')&excluded.k.eq(8)].cell.tolist()
            assert b==sb,('excluded allocation',f.fold,fish)
            z=evaluate(t,a,b,othercells);z['fold']=f.fold;base.append(z)
            mask=keys.fish.eq(fish).to_numpy(); fk=keys[mask].reset_index(drop=True); fa=ff[mask];fo=orig[mask]
            wanted=len(groups[fish]); valid_reps=0
            for rep in range(REPS):
                keep={name:np.ones(len(m),bool) for name,m in groups.items()}
                options=[(name,i) for name,m in groups.items() for i in range(len(m))]
                deleted=0
                for j in RNG.permutation(len(options)):
                    name,i=options[j]; keep[name][i]=False
                    if ((groups[name][keep[name]].sum(axis=2)>0).sum(axis=0)<3).any():keep[name][i]=True;continue
                    deleted+=1
                    if deleted==wanted:break
                if deleted!=wanted:continue
                dp=from_matrices(groups,{name:np.flatnonzero(k) for name,k in keep.items()})
                dcells=[c for c,pv in zip(cells,dp) if pv>0]
                if len(dcells)<8:continue
                chosen=rank(dp,cells);cov=fa[:,np.isin(future_cells,chosen)].sum(axis=1)
                zz=fk.copy();zz['original']=fo;zz['coverage']=cov;zz['net_loss']=fo-cov;zz['fold']=f.fold;zz['rep']=rep
                zz['changed']=len(set(a)-set(chosen)); controls.append(zz);valid_reps+=1
            sorted_p=np.sort(p)[::-1]
            support.append(dict(fold=f.fold,fish=fish,deleted_blocks=wanted,valid_reps=valid_reps,refit_rejected=rejected,all_blocks=sum(len(m) for m in groups.values()),changed=len(set(a)-set(b)),rank_margin=sorted_p[7]-sorted_p[8]))
        print('reservoir fold',f.fold,flush=True)
    base=pd.concat(base,ignore_index=True);boot=pd.concat(boot,ignore_index=True);controls=pd.concat(controls,ignore_index=True)
    assert len(base)==458 and base.fish.nunique()==5
    assert (boot.groupby('rep').size()==458).all()
    assert (controls.groupby('rep').size()==458).all()
    saved=pd.read_csv(BASE/'Review reinforcement v42 1010/Telemetry/Opportunity components per cycle 1010.csv')
    saved=saved[saved.method.eq('Pooled hotspot')&saved.k.eq(8)]
    merged=base.merge(saved,on=['fish','cycle','phase','fold'],validate='one_to_one')
    np.testing.assert_allclose(merged.coverage,merged.heldout_coverage,atol=1e-14)
    save(base,'Reservoir exclusion days');save(boot,'Reservoir refit days');save(controls,'Reservoir deletion days');save(pd.DataFrame(support),'Reservoir control support')
    metrics=['original','coverage','net_loss']
    summaries={}
    for label,z in [('exclusion',base),('refit',boot),('deletion',controls)]:
        keys=['phase'] if label=='exclusion' else ['rep','phase']
        q=z.groupby(keys+['fish'],as_index=False)[metrics].mean().groupby(keys,as_index=False)[metrics].mean()
        save(q,'Reservoir '+label+' summary');summaries[label]=q.to_dict('records') if label=='exclusion' else q.groupby('phase').net_loss.quantile([.025,.5,.975]).to_dict()
    print(str(summaries),flush=True)

if __name__=='__main__':
    marine();reservoir()
    inputs=[V25/'outputs/Primary sampled positions.parquet',V25/'outputs/Fold support primary.csv',V28/'public assets/NewmanSound_year1.csv',V28/'validation/external/outputs/Fold support.csv',Path(__file__).with_name('protocol_20261021.txt')]
    audit={'status':'PASS','seed':20261021,'repetitions':REPS,'python':sys.version,'numpy':np.__version__,'pandas':pd.__version__,'inputs':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}}
    (OUT/'audit_20261021.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
