#  _____ _____ ____   ____
# |  ___|  ___|  _ \ / ___|
# | |_  | |_  | |_) | |
# |  _| |  _| |  _ <| |___
# |_|   |_|   |_| \_\____|
# FFRC
"""分析与复算程序；原始说明及来源见来源档案。"""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import hashlib
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
SEED = 20261008

def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def p_none(n, m, k):
    """Exact hypergeometric no-positive probability, evaluated by stable recurrence."""
    k = np.asarray(k, dtype=float)
    out = np.ones_like(k)
    for j in range(int(m)):
        out *= np.maximum(n - k - j, 0) / (n - j)
    return out

def global_expected(scores, positive, groups, budgets):
    budgets = np.asarray(budgets, dtype=int)
    result = np.zeros((len(budgets), 2))
    covered = set(); prefix = 0; hit_prefix = 0
    for score in np.unique(scores)[::-1]:
        idx = np.flatnonzero(scores == score); n = len(idx)
        active = (budgets > prefix) & (budgets <= prefix + n)
        if active.any():
            q = budgets[active] - prefix
            result[active, 0] = hit_prefix + q / n * positive[idx].sum()
            result[active, 1] = len(covered)
            gp, amounts = np.unique(groups[idx[positive[idx]]], return_counts=True)
            for g, m in zip(gp, amounts):
                if g not in covered:
                    result[active, 1] += 1 - p_none(n, m, q)
        covered.update(groups[idx[positive[idx]]].tolist())
        hit_prefix += positive[idx].sum(); prefix += n
    if np.any(budgets < 1) or np.any(budgets > len(scores)):
        raise ValueError('Budget outside finite pool')
    return result

def grouped_expected(scores, positive, groups, budgets, random_heads=False):
    peaks=[]; probabilities=[]
    for g in np.unique(groups):
        idx=np.flatnonzero(groups==g)
        heads=idx if random_heads else idx[scores[idx]==scores[idx].max()]
        peaks.append(0 if random_heads else scores[idx].max())
        probabilities.append(positive[heads].mean())
    peaks=np.asarray(peaks); probabilities=np.asarray(probabilities)
    out=np.zeros((len(budgets),2)); prefix=0; cumulative=0.
    budgets=np.asarray(budgets)
    for score in np.unique(peaks)[::-1]:
        probs=probabilities[peaks==score]; n=len(probs)
        active=(budgets>prefix)&(budgets<=prefix+n)
        v=cumulative+(budgets[active]-prefix)/n*probs.sum()
        out[active,0]=v; out[active,1]=v
        cumulative+=probs.sum(); prefix+=n
    if np.any(budgets<1) or np.any(budgets>len(peaks)):
        raise ValueError('Grouped strategy limited to first round')
    return out

def ious(box, refs):
    if not len(refs): return np.empty(0)
    lo=np.maximum(box[:2],refs[:,:2]); hi=np.minimum(box[2:],refs[:,2:])
    inter=np.maximum(hi-lo,0).prod(axis=1)
    union=np.maximum(box[2:]-box[:2],0).prod()+np.maximum(refs[:,2:]-refs[:,:2],0).prod(axis=1)-inter
    return np.divide(inter,union,out=np.zeros_like(inter),where=union>0)

def match(predictions, fish, uncertain, threshold, merge=False):
    refs=np.concatenate([fish,uncertain]) if merge else fish
    used=np.zeros(len(refs),dtype=bool); tp=fp=ignored=0
    # 置信度降序，一对一；同置信度保留原预测次序。
    for pred in sorted(predictions,key=lambda p:-p['confidence']):
        overlap=ious(np.asarray(pred['bbox'],dtype=float),refs)
        overlap[used]=-1
        j=int(overlap.argmax()) if len(overlap) else -1
        if j>=0 and overlap[j]>=threshold:
            used[j]=True; tp+=1
        elif not merge and np.any(ious(np.asarray(pred['bbox'],dtype=float),uncertain)>=threshold):
            ignored+=1
        else: fp+=1
    return tp,fp,len(refs)-tp,ignored

def metrics(tp,fp,fn):
    return dict(precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None)

def load_inputs(base):
    paths=[base/'Input/COCO annotation reference.json',base/'Input/Reference frame ledger.csv',base/'Frame predictions.jsonl']
    coco=json.loads(paths[0].read_text())
    with paths[1].open() as f: ledger=list(csv.DictReader(f))
    # 未完成或末行不完整的预测文件不产出正式结果。
    predictions=[json.loads(line) for line in paths[2].read_text().splitlines() if line.strip()]
    image_ids={im['id'] for im in coco['images']}
    image_info={im['id']:im for im in coco['images']}
    lids=[int(r['image_id']) for r in ledger]; pids=[p['image_id'] for p in predictions]
    if len(coco['images'])!=2040 or len(image_ids)!=2040 or len(lids)!=2040 or len(pids)!=2040:
        raise ValueError(f'Need complete 2040-frame pool; COCO={len(image_ids)}, ledger={len(lids)}, predictions={len(pids)}')
    if len(set(lids))!=len(lids) or len(set(pids))!=len(pids) or set(lids)!=image_ids or set(pids)!=image_ids:
        raise ValueError('Duplicate or mismatched image IDs')
    ledger=sorted(ledger,key=lambda r:int(r['image_id'])); pmap={p['image_id']:p for p in predictions}
    predictions=[pmap[int(r['image_id'])] for r in ledger]
    ref=defaultdict(lambda:defaultdict(list))
    for a in coco['annotations']:
        x,y,w,h=a['bbox']; ref[a['image_id']][a['category_id']].append([x,y,x+w,y+h])
    videos=sorted({r['source_video'] for r in ledger}); vmap={v:i for i,v in enumerate(videos)}
    if len(videos)!=34: raise ValueError('Need 34 source videos')
    groups=np.array([vmap[r['source_video']] for r in ledger])
    fish=[]; uncertain=[]
    for r,p in zip(ledger,predictions):
        iid=int(r['image_id'])
        f=np.asarray(ref[iid][1],dtype=float).reshape(-1,4); u=np.asarray(ref[iid][3],dtype=float).reshape(-1,4)
        if len(f)!=int(r['fish_boxes']) or len(u)!=int(r['uncertain_boxes']): raise ValueError('Ledger/reference disagreement')
        if p['dimensions']!=[image_info[iid]['width'],image_info[iid]['height']]: raise ValueError('Prediction/reference dimensions disagree')
        if p['segment']!=r['segment'] or int(p['frame'])!=int(r['frame']) or int(p['task_id'])!=int(r['task_id']): raise ValueError('Prediction provenance mismatch')
        for method in ['static','bsca']:
            for det in p[method]:
                b=np.asarray(det['bbox'],float)
                if b.shape!=(4,) or not np.isfinite(b).all() or np.any(b[2:]<=b[:2]) or not math.isfinite(det['confidence']): raise ValueError('Invalid prediction')
        fish.append(f); uncertain.append(u)
    return paths,ledger,predictions,fish,uncertain,groups,videos

def review_rows(scores, positive, groups, subset):
    n=len(positive); g=len(np.unique(groups)); pf=int(positive.sum()); pv=len(np.unique(groups[positive]))
    rows=[]
    strategies={}
    for m,s in scores.items():
        strategies[f'{m}_global']=global_expected(s,positive,groups,np.arange(1,n+1))
        strategies[f'{m}_grouped_first_round']=grouped_expected(s,positive,groups,np.arange(1,g+1))
    strategies['uniform_frame_random']=global_expected(np.zeros(n),positive,groups,np.arange(1,n+1))
    strategies['random_one_frame_per_video']=grouped_expected(np.zeros(n),positive,groups,np.arange(1,g+1),True)
    for name,curve in strategies.items():
        if np.any(np.diff(curve,axis=0)<-1e-9): raise AssertionError('Review frontier must be monotone')
        if len(curve)==n and name not in ['random_one_frame_per_video'] and not name.endswith('grouped_first_round'):
            if not np.allclose(curve[-1],[pf,pv]): raise AssertionError('Full-pool endpoint mismatch')
        for k,(fh,vh) in enumerate(curve,1):
            increment=vh-(curve[k-2,1] if k>1 else 0.)
            rows.append(dict(reference_rule=subset,strategy=name,budget=k,expected_new_positive_videos_at_this_frame=float(increment),expected_positive_videos_per_reviewed_frame=float(vh/k),expected_positive_frames=float(fh),positive_frame_recall=float(fh/pf) if pf else None,expected_positive_videos=float(vh),positive_video_recall=float(vh/pv) if pv else None,eligible_frames=n,reference_positive_frames=pf,eligible_videos=g,reference_positive_videos=pv))
    return rows

def bootstrap(scores,positive,groups,repetitions):
    rng=np.random.default_rng(SEED); unique=np.unique(groups); members=[np.flatnonzero(groups==g) for g in unique]
    n=len(positive); points={}
    for k in [34,min(70,n)]:
        points[('bsca_global_minus_static_global',k)]=global_expected(scores['bsca'],positive,groups,[k])[0]-global_expected(scores['static'],positive,groups,[k])[0]
    for m in scores:
        points[(f'{m}_grouped_minus_global',34)]=grouped_expected(scores[m],positive,groups,[34])[0]-global_expected(scores[m],positive,groups,[34])[0]
    random_point=grouped_expected(np.zeros(n),positive,groups,[34],True)[0]
    for m in scores:
        points[(f'{m}_grouped_minus_random_one_frame_per_video',34)]=grouped_expected(scores[m],positive,groups,[34])[0]-random_point
    samples={key:[] for key in points}
    for _ in range(repetitions):
        draw=rng.integers(0,len(unique),size=len(unique))
        idx=np.concatenate([members[j] for j in draw])
        # 重采样的重复视频当作各自完整的来源簇副本。
        bg=np.concatenate([np.full(len(members[j]),i) for i,j in enumerate(draw)])
        bp=positive[idx]; bs={m:s[idx] for m,s in scores.items()}
        globals_={m:global_expected(s,bp,bg,[34,min(70,len(idx))]) for m,s in bs.items()}
        for j,k in enumerate([34,min(70,n)]): samples[('bsca_global_minus_static_global',k)].append(globals_['bsca'][j]-globals_['static'][j])
        random_sample=grouped_expected(np.zeros(len(idx)),bp,bg,[34],True)[0]
        for m in bs:
            grouped_sample=grouped_expected(bs[m],bp,bg,[34])[0]
            samples[(f'{m}_grouped_minus_global',34)].append(grouped_sample-globals_[m][0])
            samples[(f'{m}_grouped_minus_random_one_frame_per_video',34)].append(grouped_sample-random_sample)
    rows=[]
    for key,vals in samples.items():
        arr=np.asarray(vals)
        for j,endpoint in enumerate(['expected_positive_frames','expected_positive_videos']):
            lo,hi=np.quantile(arr[:,j],[.025,.975])
            rows.append(dict(comparison=key[0],budget=key[1],endpoint=endpoint,estimate=float(points[key][j]),ci_low=float(lo),ci_high=float(hi),bootstrap_repetitions=repetitions,seed=SEED,unit='paired source-video cluster; duplicates retained as separate cluster copies',interpretation='conditional on observed selected video segments; not population guarantee'))
    return rows

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--base',type=Path,default=HERE); parser.add_argument('--bootstrap-repetitions',type=int,default=3000)
    args=parser.parse_args()
    if args.bootstrap_repetitions!=3000: raise ValueError('Formal evaluation requires 3000 replicates')
    paths,ledger,preds,fish,uncertain,groups,videos=load_inputs(args.base)
    out=args.base/'Evaluation'; out.mkdir(exist_ok=True)
    positive=np.array([len(f)>0 for f in fish]); has_uncertain=np.array([len(u)>0 for u in uncertain])
    scores={m:np.array([len(p[m]) for p in preds]) for m in ['static','bsca']}
    frame_rows=[]; object_rows=[]; video_rows=[]
    for rule in ['fish_ignore_uncertain','fish_plus_uncertain','exclude_uncertain_frames']:
        keep=~has_uncertain if rule=='exclude_uncertain_frames' else np.ones(len(preds),bool)
        for threshold in [.30,.50]:
            for method in scores:
                records=[]
                for i in np.flatnonzero(keep):
                    tp,fp,fn,ignored=match(preds[i][method],fish[i],uncertain[i],threshold,rule=='fish_plus_uncertain')
                    records.append((i,tp,fp,fn,ignored))
                    frame_rows.append(dict(image_id=int(ledger[i]['image_id']),source_video=videos[groups[i]],reference_rule=rule,iou=threshold,method=method,fish_boxes=len(fish[i]),uncertain_boxes=len(uncertain[i]),prediction_count=int(scores[method][i]),tp=tp,fp=fp,fn=fn,ignored_predictions=ignored))
                a=np.asarray(records); totals=a[:,1:].sum(axis=0); tp,fp,fn,ignored=map(int,totals)
                r=dict(reference_rule=rule,iou=threshold,method=method,eligible_frames=int(keep.sum()),reference_objects=tp+fn,tp=tp,fp=fp,fn=fn,ignored_predictions=ignored,**metrics(tp,fp,fn))
                refpos=np.array([(len(fish[i])+ (len(uncertain[i]) if rule=='fish_plus_uncertain' else 0))>0 for i in np.flatnonzero(keep)])
                matched=a[:,1]>0
                r.update(reference_positive_frames=int(refpos.sum()),matched_positive_frames=int(matched.sum()),frame_reference_detection_recall=float(matched.sum()/refpos.sum()) if refpos.sum() else None,reference_positive_videos=len(set(groups[a[refpos,0]])),matched_positive_videos=len(set(groups[a[matched,0]])))
                r['video_reference_detection_recall']=r['matched_positive_videos']/r['reference_positive_videos'] if r['reference_positive_videos'] else None
                object_rows.append(r)
                for g in np.unique(groups[keep]):
                    v=a[groups[a[:,0]]==g]; vt,vf,vn,vi=map(int,v[:,1:].sum(axis=0))
                    video_rows.append(dict(reference_rule=rule,iou=threshold,method=method,source_video=videos[g],frames=len(v),tp=vt,fp=vf,fn=vn,ignored_predictions=vi,**metrics(vt,vf,vn)))
    review=[]
    for rule,pos,keep in [('fish',positive,np.ones(len(preds),bool)),('fish_plus_uncertain',positive|has_uncertain,np.ones(len(preds),bool)),('exclude_uncertain_frames',positive,~has_uncertain)]:
        review.extend(review_rows({m:s[keep] for m,s in scores.items()},pos[keep],groups[keep],rule))
    alert=[]
    for m,s in scores.items():
        keep=s>0
        if keep.any():
            alert.extend(review_rows({m:s[keep]},positive[keep],groups[keep],f'fish_alert_only_{m}'))
    for r in alert:
        r.update(full_pool_positive_frames=int(positive.sum()),full_pool_positive_videos=len(np.unique(groups[positive])),full_pool_positive_frame_recall=r['expected_positive_frames']/positive.sum(),full_pool_positive_video_recall=r['expected_positive_videos']/len(np.unique(groups[positive])))
    boot=bootstrap(scores,positive,groups,args.bootstrap_repetitions)
    write_csv(out/'Object metrics.csv',object_rows); write_csv(out/'Frame matching.csv',frame_rows); write_csv(out/'Video object metrics.csv',video_rows)
    write_csv(out/'Review budget curves.csv',review); write_csv(out/'Alert-only budget curves.csv',alert); write_csv(out/'Paired video bootstrap.csv',boot)
    # 固定seed队列是可执行实例；正式曲线使用并列的解析期望。
    queue=[]
    for m,s in scores.items():
        rng=np.random.default_rng(SEED)
        order=np.lexsort((rng.random(len(s)),-s))
        for rank,i in enumerate(order,1): queue.append(dict(strategy=f'{m}_global',rank=rank,image_id=int(ledger[i]['image_id']),source_video=videos[groups[i]],predicted_count=int(s[i]),seed=SEED))
        heads=[]
        for g in np.unique(groups):
            idx=np.flatnonzero(groups==g); heads.append(int(rng.choice(idx[s[idx]==s[idx].max()])))
        heads=np.array(heads); heads=heads[np.lexsort((rng.random(len(heads)),-s[heads]))]
        for rank,i in enumerate(heads,1): queue.append(dict(strategy=f'{m}_grouped_first_round',rank=rank,image_id=int(ledger[i]['image_id']),source_video=videos[groups[i]],predicted_count=int(s[i]),seed=SEED))
    rng=np.random.default_rng(SEED)
    for rank,i in enumerate(rng.permutation(len(preds)),1):
        queue.append(dict(strategy='uniform_frame_random',rank=rank,image_id=int(ledger[i]['image_id']),source_video=videos[groups[i]],predicted_count='',seed=SEED))
    for rank,g in enumerate(rng.permutation(np.unique(groups)),1):
        i=int(rng.choice(np.flatnonzero(groups==g)))
        queue.append(dict(strategy='random_one_frame_per_video',rank=rank,image_id=int(ledger[i]['image_id']),source_video=videos[g],predicted_count='',seed=SEED))
    write_csv(out/'Fixed seed review queues.csv',queue)
    census=[]
    for g,v in enumerate(videos):
        idx=np.flatnonzero(groups==g)
        census.append(dict(source_video=v,segment=ledger[idx[0]]['segment'],source_start=ledger[idx[0]]['source_start'],batch=ledger[idx[0]]['batch'],scene=ledger[idx[0]]['scene'],frames=len(idx),fish_objects=sum(len(fish[i]) for i in idx),uncertain_objects=sum(len(uncertain[i]) for i in idx),fish_positive_frames=int(positive[idx].sum()),uncertain_frames=int(has_uncertain[idx].sum()),static_prediction_count=int(scores['static'][idx].sum()),bsca_prediction_count=int(scores['bsca'][idx].sum())))
    write_csv(out/'Video reference and prediction census.csv',census)
    summary=dict(status='complete',frames=len(preds),source_videos=len(videos),reference_fish_objects=sum(map(len,fish)),reference_uncertain_objects=sum(map(len,uncertain)),reference_fish_positive_frames=int(positive.sum()),reference_fish_positive_videos=len(np.unique(groups[positive])),uncertain_frames=int(has_uncertain.sum()),prediction_counts={m:int(s.sum()) for m,s in scores.items()},definitions=dict(object_matching='Confidence-descending greedy one-to-one maximum-IoU available reference; unmatched predictions with uncertain IoU >= threshold ignored for primary rule. COCO xywh converted to xyxy; prediction xyxy.',primary_reference='fish category 1; uncertain category 3',review_success='Reviewed frame contains reference fish; video success requires selecting at least one reference-positive frame in that source video. GT never sorts queues.',global_queue='Predicted-count descending over all frames; zero-score frames remain eligible. This differs from operational alert-only selection.',ties='Uniform choice without replacement at global cutoff; exact P(no hit)=C(n-m,k)/C(n,k). Group video score=max predicted count; uniform top-frame ties within video and uniform cutoff video ties.',random_video_baseline='Uniform video subset, then uniform one frame within each selected video; first round only.',grouping='Source video is an acquisition segment, not a geographically independent site.',review_iou='Queue review success depends on manual reference presence, not IoU; object and automatic frame detection endpoints separately use IoU 0.30 and 0.50.',bootstrap='3000 paired video-cluster replicates seed 20261008; all frames retained for each sampled cluster; duplicate sampled videos represented as separate cluster copies; percentile 95% intervals of count endpoint differences conditional on selected sample.'),evaluator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),numpy_version=np.__version__,sources={str(p.relative_to(args.base)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},object_metrics=object_rows,workload_points=[r for r in review if r['reference_rule']=='fish' and r['budget'] in [34,70]],paired_bootstrap=boot)
    summary_tmp=out/'Evaluation summary.json.tmp'
    summary_tmp.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    summary_tmp.replace(out/'Evaluation summary.json')
    print(json.dumps({k:summary[k] for k in ['status','frames','source_videos','reference_fish_objects','reference_fish_positive_frames','reference_fish_positive_videos','prediction_counts']},ensure_ascii=False))

if __name__=='__main__': main()
