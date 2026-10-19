#  _____ _____ ____   ____
# |  ___|  ___|  _ \ / ___|
# | |_  | |_  | |_) | |
# |  _| |  _| |  _ <| |___
# |_|   |_|   |_| \_\____|
# FFRC
"""从冻结选格和位置记录分解留鱼覆盖差；不重新拟合或增加样本。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'submission/Ecological Indicators/Reproducibility/analysis'
OUT=ROOT/'analysis/innovation reframing 20261009 v33'
if (Path(__file__).resolve().parent.parent/'reservoir ecology v25').is_dir():
    OUT=Path(__file__).resolve().parent
    BASE=OUT.parent
    ROOT=OUT.parents[2]
METHODS=['Day hotspot','Pooled hotspot','State balance']
BUDGETS=[1,2,4,8,16,32]
METRICS=['familiar_coverage','heldout_coverage','net_loss','outside_other_history_loss',
         'shared_history_loss','heldout_configuration_gain','retained_coverage',
         'outside_familiar_candidate','outside_other_candidate']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    paths={
        'positions':BASE/'reservoir ecology v25/outputs/Primary sampled positions.parquet',
        'folds':BASE/'reservoir ecology v25/outputs/Fold support primary.csv',
        'familiar_allocations':BASE/'reservoir ecology v25/outputs/Training allocations.csv',
        'heldout_allocations':BASE/'reservoir innovation v28/validation/outputs/Validation allocations.csv',
        'future_scores':BASE/'reservoir innovation v28/validation/outputs/Fish cycle coverage.csv',
        'support_audits':BASE/'reservoir innovation v28/validation/outputs/Support audits.csv'}
    s=pd.read_parquet(paths['positions']);s['cycle']=s.cycle.astype(str)
    folds=pd.read_csv(paths['folds']);a=pd.read_csv(paths['familiar_allocations'])
    b=pd.read_csv(paths['heldout_allocations']);scores=pd.read_csv(paths['future_scores'])
    audits=pd.read_csv(paths['support_audits'])
    records=[];candidate_checks=[]
    for f in folds.itertuples():
        train=s[(s.cycle>=f.train_start)&(s.cycle<f.test_start)]
        familiar_ids=f.training_fish.split('|')
        familiar_candidates=set(train[train.fish.isin(familiar_ids)].cell)
        future=s[(s.cycle>=f.test_start)&(s.cycle<f.test_end_exclusive)&s.fish.isin(familiar_ids)]
        paired=future.groupby(['fish','cycle']).phase.nunique().eq(2)
        future=future.join(paired.rename('_paired'),on=['fish','cycle']).query('_paired').drop(columns='_paired')
        for fish,t in future.groupby('fish'):
            other_train=train[train.fish!=fish]
            n=other_train[['fish','cycle','phase']].drop_duplicates().groupby(['fish','phase']).size().unstack(fill_value=0)
            other_ids=sorted(n.index[(n.Day>=3)&(n.Night>=3)].tolist())
            other_candidates=set(other_train[other_train.fish.isin(other_ids)].cell)
            saved=audits[(audits.design=='B')&(audits.fold==f.fold)&(audits.heldout==fish)].iloc[0]
            assert set(saved.training_ids.split('|'))==set(other_ids)
            assert len(other_candidates)==saved.candidate_cells
            assert fish not in other_ids
            assert other_candidates.issubset(familiar_candidates),'Candidate histories are not nested'
            candidate_checks.append(dict(fold=f.fold,fish=fish,other_training_ids='|'.join(other_ids),
                                         familiar_candidate_cells=len(familiar_candidates),other_candidate_cells=len(other_candidates)))
            in_other=t.cell.isin(other_candidates).to_numpy()
            for method in METHODS:
                for k in BUDGETS:
                    chosen_a=set(a[(a.fold==f.fold)&(a.method==method)&(a.k==k)].cell)
                    chosen_b=set(b[(b.design=='B')&(b.fold==f.fold)&(b.heldout==fish)&(b.method==method)&(b.k==k)].cell)
                    assert len(chosen_a)==len(chosen_b)==k
                    assert chosen_a.issubset(familiar_candidates) and chosen_b.issubset(other_candidates)
                    ca=t.cell.isin(chosen_a).to_numpy();cb=t.cell.isin(chosen_b).to_numpy()
                    lost=ca&~cb;gained=~ca&cb
                    z=t[['fish','cycle','phase']].copy()
                    z['familiar_coverage']=ca.astype(float);z['heldout_coverage']=cb.astype(float)
                    z['net_loss']=ca.astype(float)-cb.astype(float)
                    z['outside_other_history_loss']=(lost&~in_other).astype(float)
                    z['shared_history_loss']=(lost&in_other).astype(float)
                    z['heldout_configuration_gain']=gained.astype(float)
                    z['retained_coverage']=(ca&cb).astype(float)
                    z['outside_familiar_candidate']=(~t.cell.isin(familiar_candidates)).to_numpy().astype(float)
                    z['outside_other_candidate']=(~in_other).astype(float)
                    # 各项互斥，逐位置差值已经满足精确恒等式。
                    assert np.array_equal(z.net_loss.to_numpy(),
                                          (z.outside_other_history_loss+z.shared_history_loss-z.heldout_configuration_gain).to_numpy())
                    q=z.groupby(['fish','cycle','phase'],as_index=False).agg(
                        **{name:(name,'mean') for name in METRICS},sample_n=('familiar_coverage','size'))
                    q['fold']=f.fold;q['method']=method;q['k']=k
                    records.append(q)
    per_cycle=pd.concat(records,ignore_index=True)
    per_cycle.to_csv(OUT/'Transfer components per cycle.csv',index=False)
    keys=['fold','fish','cycle','phase','method','k']
    for design,column in [('A','familiar_coverage'),('B','heldout_coverage')]:
        old=scores[(scores.design==design)&(scores.cohort=='original_eligible')&scores.method.isin(METHODS)]
        joined=per_cycle.merge(old[keys+['coverage','sample_n','outside_candidate']],on=keys,suffixes=('','_saved'),validate='one_to_one')
        assert len(joined)==len(per_cycle)==len(old)
        np.testing.assert_allclose(joined[column],joined.coverage,atol=1e-14,rtol=0)
        assert (joined.sample_n==joined.sample_n_saved).all()
        outside='outside_familiar_candidate' if design=='A' else 'outside_other_candidate'
        np.testing.assert_allclose(joined[outside],joined.outside_candidate,atol=1e-14,rtol=0)
    fish=per_cycle.groupby(['method','k','phase','fish'],as_index=False).agg(
        **{name:(name,'mean') for name in METRICS},paired_cycles=('cycle','size'))
    fish.to_csv(OUT/'Transfer components per fish.csv',index=False)
    summary=fish.groupby(['method','k','phase'],as_index=False).agg(
        **{name:(name,'mean') for name in METRICS},fish=('fish','nunique'),paired_cycles=('paired_cycles','sum'))
    # 比例仅表示正的净差额构成。分项可大于净额，不是概率或因果份额。
    summary['shared_history_share_of_gross_loss']=summary.shared_history_loss/(summary.shared_history_loss+summary.outside_other_history_loss)
    summary['outside_history_share_of_gross_loss']=1-summary.shared_history_share_of_gross_loss
    summary['fractional_net_loss']=summary.net_loss/summary.familiar_coverage
    summary.to_csv(OUT/'Transfer components summary.csv',index=False)
    pd.DataFrame(candidate_checks).to_csv(OUT/'Candidate history checks.csv',index=False)
    np.testing.assert_allclose(summary.net_loss,summary.outside_other_history_loss+summary.shared_history_loss-summary.heldout_configuration_gain,atol=1e-14,rtol=0)
    pooled=summary[(summary.method=='Pooled hotspot')&(summary.k==8)&(summary.phase=='Night')].iloc[0].to_dict()
    assert pooled['fish']==5 and pooled['paired_cycles']==229
    primary=per_cycle[(per_cycle.method=='Pooled hotspot')&(per_cycle.k==8)]
    assert len(primary)==458
    supports=primary.pivot(index=['fish','cycle'],columns='phase',values='sample_n')
    sensitivity=[]
    for threshold in [12,24,36,48,60]:
        eligible=supports[(supports.Day>=threshold)&(supports.Night>=threshold)].index
        q=primary[primary.phase.eq('Night')].set_index(['fish','cycle']).loc[eligible].reset_index()
        fg=q.groupby('fish')[METRICS].mean();means=fg.mean()
        sensitivity.append(dict(minimum_positions_both_windows=threshold,fish=len(fg),paired_cycles=len(q),
            **{name:float(means[name]) for name in METRICS},
            shared_history_share_of_gross_loss=float(means.shared_history_loss/(means.shared_history_loss+means.outside_other_history_loss))))
    pd.DataFrame(sensitivity).to_csv(OUT/'Transfer support sensitivity.csv',index=False)
    # 构成分项不应被误称成观察外推的生态因果机制。
    report=dict(status='PASS',analysis='Exact coverage-difference decomposition under fixed historical configurations',
                primary=pooled,support_sensitivity=sensitivity,matched_cycle_window_budget_policy_rows=len(per_cycle),
                matches_existing_A_and_B_scores=True,nested_candidate_histories=True,
                arithmetic_maximum_error=float(np.max(np.abs(summary.net_loss-(summary.outside_other_history_loss+summary.shared_history_loss-summary.heldout_configuration_gain)))),
                weighting='Position fractions per fish-cycle-window, cycles within fish, equal fish means',
                scientific_boundary='Observed tagged cohort; retrospective counterfactual accounting, not ecological causal attribution or a new animal sample',
                source_files={name:dict(path=str(path.relative_to(ROOT)),sha256=sha(path)) for name,path in paths.items()})
    (OUT/'Transfer decomposition audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report['primary'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
