#  _____ _____ ____   ____
# |  ___|  ___|  _ \ / ___|
# | |_  | |_  | |_) | |
# |  _| |  _| |  _ <| |___
# |_|   |_|   |_| \_\____|
# FFRC
# ><> 1010 已完成标注任务的冻结预测复算。
"""复算标注状态敏感性，不重新推理、拟合或调整阈值。"""
from pathlib import Path
from collections import Counter
import csv, hashlib, importlib.util, json, platform, shutil
import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BASE = ROOT / 'analysis/Hangbu SAT validation 20261008'
AUDIT = ROOT / 'analysis/Hangbu CVAT audit 20261008'
REGISTRY = OUT / 'Source registry'
INPUT = OUT / 'Input'
SEED = 20261008
REPS = 3000

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open() as f: return list(csv.DictReader(f))
def save(name, rows):
    with (OUT/name).open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def js(name, value): (OUT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')

def bootstrap(ev,scores,pos,groups,k):
    rng=np.random.default_rng(SEED)
    members=[np.flatnonzero(groups==g) for g in np.unique(groups)]
    def effects(s,p,g):
        gl={m:ev.global_expected(v,p,g,[k])[0] for m,v in s.items()}
        gr={m:ev.grouped_expected(v,p,g,[k])[0] for m,v in s.items()}
        ra=ev.grouped_expected(np.zeros(len(p)),p,g,[k],True)[0]
        z={'bsca_global_minus_static_global':gl['bsca']-gl['static']}
        for m in s:
            z[m+'_grouped_minus_global']=gr[m]-gl[m]
            z[m+'_grouped_minus_random_one_frame_per_video']=gr[m]-ra
        return z
    estimates=effects(scores,pos,groups); samples={x:[] for x in estimates}
    for _ in range(REPS):
        draw=rng.integers(0,len(members),size=len(members))
        idx=np.concatenate([members[j] for j in draw])
        # 抽到的重复视频各保留60帧，分别作为来源簇副本。
        bg=np.concatenate([np.full(len(members[j]),i) for i,j in enumerate(draw)])
        z=effects({m:s[idx] for m,s in scores.items()},pos[idx],bg)
        for key,value in z.items(): samples[key].append(value)
    rows=[]
    for key,values in samples.items():
        a=np.array(values)
        for j,endpoint in enumerate(['expected_positive_frames','expected_positive_videos']):
            lo,hi=np.quantile(a[:,j],[.025,.975])
            rows.append(dict(comparison=key,budget=k,endpoint=endpoint,estimate=float(estimates[key][j]),ci_low=float(lo),ci_high=float(hi),bootstrap_repetitions=REPS,seed=SEED,cluster_count=k,unit='source-video clusters; duplicated draws retained as distinct complete copies',interpretation='percentile interval conditional on recorded video sources, not independently sampled sites'))
    return rows

def main():
    OUT.mkdir(parents=True,exist_ok=True); INPUT.mkdir(exist_ok=True)
    source_map={}
    for src,dest in [(BASE/'evaluate_predictions_20261019.py',INPUT/'Frozen evaluation functions 1010.py'),(BASE/'Input/COCO annotation reference.json',INPUT/'COCO annotation reference.json'),(BASE/'Input/Reference frame ledger.csv',INPUT/'Reference frame ledger.csv'),(BASE/'Frame predictions.jsonl',INPUT/'Frame predictions.jsonl'),(AUDIT/'jobs live.json',INPUT/'Jobs export 1008.json'),(AUDIT/'tasks live.json',INPUT/'Tasks export 1008.json'),(AUDIT/'Original video provenance.csv',INPUT/'Original video provenance.csv'),(AUDIT/'Live audit summary.json',INPUT/'CVAT audit summary 1008.json'),(REGISTRY/'segments_v1.json',INPUT/'Base source registry 1010.json'),(REGISTRY/'segments_v1_ext.json',INPUT/'Extension source registry 1010.json')]:
        if not dest.exists(): shutil.copy2(src,dest)
        if src.exists() and sha(src)!=sha(dest): raise ValueError('Frozen input differs from source: '+str(src))
        source_map[str(dest.relative_to(OUT))]=dict(original_path=str(src.relative_to(ROOT)) if src.is_relative_to(ROOT) else str(src),sha256=sha(dest))
    # 模块加载仅定义函数，不执行原34段主评价。
    spec=importlib.util.spec_from_file_location('frozen_evaluation',INPUT/'Frozen evaluation functions 1010.py')
    ev=importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    (INPUT/'Input').mkdir(exist_ok=True)
    # 原加载器保留完整2040帧一致性检查；别名在本输出目录内。
    for name in ['COCO annotation reference.json','Reference frame ledger.csv']:
        alias=INPUT/'Input'/name
        if not alias.exists(): shutil.copy2(INPUT/name,alias)
        assert sha(alias)==sha(INPUT/name)
    paths,ledger,preds,fish,uncertain,groups,videos=ev.load_inputs(INPUT)
    jobs=json.loads((INPUT/'Jobs export 1008.json').read_text())
    tasks={int(t['id']):t for t in json.loads((INPUT/'Tasks export 1008.json').read_text())}
    registry={int(r['task_id']):r for r in read(INPUT/'Original video provenance.csv')}
    raw_registry={i:r for i,r in enumerate(json.loads((INPUT/'Base source registry 1010.json').read_text()))}
    raw_registry.update({r['idx']:r for r in json.loads((INPUT/'Extension source registry 1010.json').read_text())})
    bytask={}
    for j in jobs:
        if j['type']!='annotation': continue
        bytask.setdefault(int(j['task_id']),[]).append(j)
    mapping=[]
    for tid in sorted({int(r['task_id']) for r in ledger}):
        rr=[r for r in ledger if int(r['task_id'])==tid]; jj=bytask[tid]; t=tasks[tid]; reg=registry[tid]
        if len(jj)!=1 or len(rr)!=60: raise ValueError('Unexpected task/job/frame mapping')
        j=jj[0]
        assert t['name']==j['task_name']=='seg_'+rr[0]['segment']
        assert reg['segment']==rr[0]['segment'] and reg['source_video']==rr[0]['source_video']
        source=raw_registry[int(reg['segment'].split('_')[0])]
        for field,rawfield in [('batch','batch'),('scene','scene'),('source_video','relpath'),('source_start','start'),('source_end','end'),('source_camera_point_code','point')]:
            assert reg[field]==source[rawfield], 'Raw registry/provenance disagreement'
        assert j['frame_count']==60 and j['start_frame']==0 and j['stop_frame']==59
        if j['state'] not in ['completed','in progress']: raise ValueError('Unexpected job state')
        mapping.append(dict(**reg,job_id=j['id'],job_state=j['state'],job_status=j['status'],job_stage=j['stage'],job_updated_date=j['updated_date'],task_status=t['status'],included_completed_subset=j['state']=='completed',frames=len(rr),fish_positive_frames=sum(int(r['fish_boxes'])>0 for r in rr),fish_boxes=sum(int(r['fish_boxes']) for r in rr),uncertain_boxes=sum(int(r['uncertain_boxes']) for r in rr),positive_video=any(int(r['fish_boxes'])>0 for r in rr),status_source=source_map['Input/Jobs export 1008.json']['original_path'],registry_source=source_map['Input/Original video provenance.csv']['original_path']))
    save('Task status and source mapping 1010.csv',mapping)
    completed={int(r['task_id']) for r in mapping if r['included_completed_subset']}
    assert len(completed)==29 and len(mapping)==34
    keep=np.array([int(r['task_id']) in completed for r in ledger]); idx=np.flatnonzero(keep)
    pos=np.array([len(f)>0 for f in fish])[keep]; grp=groups[keep]
    scores={m:np.array([len(p[m]) for p in preds])[keep] for m in ['static','bsca']}
    k=len(completed)
    curves=ev.review_rows(scores,pos,grp,'fish_completed_jobs')
    for row in curves:
        row['positive_frame_yield']=row['expected_positive_frames']/row['budget']
    save('Completed all budget curves 1010.csv',curves)
    points=[r for r in curves if r['budget']==k]; save('Completed equal budget 1010.csv',points)
    boot=bootstrap(ev,scores,pos,grp,k); save('Completed paired video bootstrap 1010.csv',boot)
    objects=[]
    for iou in [.30,.50]:
        for m in scores:
            a=np.array([ev.match(preds[i][m],fish[i],uncertain[i],iou) for i in idx])
            tp,fp,fn,ignored=map(int,a.sum(axis=0))
            objects.append(dict(reference_rule='fish_ignore_uncertain',method=m,iou=iou,eligible_frames=len(idx),reference_fish_objects=sum(len(fish[i]) for i in idx),tp=tp,fp=fp,fn=fn,ignored_predictions=ignored,**ev.metrics(tp,fp,fn)))
    save('Completed object metrics 1010.csv',objects)
    counts={}
    for label,included in [('included_completed',True),('excluded_in_progress',False)]:
        rows=[r for r in mapping if r['included_completed_subset']==included]
        counts[label]=dict(videos=len(rows),frames=sum(r['frames'] for r in rows),positive_videos=sum(r['positive_video'] for r in rows),positive_frames=sum(r['fish_positive_frames'] for r in rows),fish_boxes=sum(r['fish_boxes'] for r in rows),uncertain_boxes=sum(r['uncertain_boxes'] for r in rows),task_ids=[int(r['task_id']) for r in rows],batch_labels=dict(Counter(r['batch'] for r in rows)),source_camera_point_codes=dict(Counter(r['source_camera_point_code'] for r in rows)),source_start_dates=dict(Counter(r['source_start'][:10] for r in rows)))
    assert counts['included_completed']['fish_boxes']+counts['excluded_in_progress']['fish_boxes']==2192
    # 全34段解析端点用于确认冻结输入与已发表在包中的数值相符。
    allpos=np.array([len(f)>0 for f in fish]); alls={m:np.array([len(p[m]) for p in preds]) for m in scores}
    original={m+'_global':ev.global_expected(s,allpos,groups,[34])[0].tolist() for m,s in alls.items()}
    original.update({m+'_grouped':ev.grouped_expected(s,allpos,groups,[34])[0].tolist() for m,s in alls.items()})
    assert np.isclose(original['bsca_global'][1],6.051153,atol=1e-6)
    assert np.isclose(original['bsca_grouped'][1],21.75)
    js('Completed sensitivity summary 1010.json',dict(status='complete',counts=counts,equal_frame_budget=k,workload_points=points,paired_bootstrap=boot,object_metrics=objects,full34_reconstruction=original,prediction_counts={m:int(s.sum()) for m,s in scores.items()},provenance=source_map,python_version=platform.python_version(),numpy_version=np.__version__,seed=SEED,bootstrap_repetitions=REPS,script_sha256=sha(Path(__file__)),boundary='Annotation workflow status sensitivity using recorded fish geometry and frozen predictions. Job completed is not an independent audit of annotation correctness. Video segments are provenance units, not independent geographic sites; batch labels are not verified acquisition dates; dates and camera codes are recorded source metadata only.'))
    print(json.dumps(dict(status='complete',counts=counts,equal_budget=points,bootstrap=boot,objects=objects),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
