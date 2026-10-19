#  _____ _____ ____   ____
# |  ___|  ___|  _ \ / ___|
# | |_  | |_  | |_) | |
# |  _| |  _| |  _ <| |___
# |_|   |_|   |_| \_\____|
# FFRC
"""监测覆盖审计：仅汇总已有覆盖，不生成布点或自动判定达标。"""
import argparse
from pathlib import Path
from html import escape
import json
import numpy as np
import pandas as pd

CONDITIONS={'zero':lambda x:x.eq(0),'below10':lambda x:x.lt(.1),'below25':lambda x:x.lt(.25),'below50':lambda x:x.lt(.5)}
def longest_daily_run(dates,condition):
    # 缺日期、缺覆盖、非目标日均打断；不把相邻文件行当连续日期。
    longest=current=0;previous=None
    for day,hit in zip(dates,condition):
        if previous is None or (day.date()-previous.date()).days!=1:current=0
        current=current+1 if bool(hit) else 0
        longest=max(longest,current);previous=day
    return longest

def audit(data,unit_col=None,date_col=None,k=None):
    unit_col=unit_col or next((c for c in ['unit','fish','source'] if c in data),None)
    date_col=date_col or next((c for c in ['date','cycle'] if c in data),None)
    if unit_col is None or date_col is None or 'coverage' not in data:raise ValueError('需要unit/fish/source、date/cycle和coverage字段')
    z=data.copy()
    if k is not None:
        if 'k' not in z:raise ValueError('--k需要输入k字段')
        z=z[pd.to_numeric(z.k,errors='coerce').eq(k)].copy()
    if z.empty:raise ValueError('筛选后无记录')
    if z[unit_col].isna().any():raise ValueError('个体/单元身份不得缺失')
    z['_date']=pd.to_datetime(z[date_col],errors='raise').dt.normalize()
    if z['_date'].isna().any():raise ValueError('日期不得缺失')
    z['_unit']=z[unit_col].astype(str)
    z['coverage']=pd.to_numeric(z.coverage,errors='raise')
    if (~z.coverage.dropna().between(0,1)).any():raise ValueError('coverage须为0..1或缺失')
    keys=[c for c in ['method','k','phase'] if c in z]
    if z[keys].isna().any().any():raise ValueError('可选分组字段不得缺失')
    if z.duplicated(keys+['_unit','_date']).any():raise ValueError('同分组、个体、日有重复：请先明确汇总队列或phase，避免重复加权')
    records=[]
    grouper=keys+['_unit']
    for values,g in z.groupby(grouper,dropna=False,sort=True):
        values=values if isinstance(values,tuple) else (values,);r=dict(zip(grouper,values));g=g.sort_values('_date');n=g.coverage.notna().sum();observed=len(g);span=(g._date.max().date()-g._date.min().date()).days+1
        r.update(mean_coverage=float(g.coverage.mean()) if n else np.nan,observed_days=observed,valid_coverage_days=int(n),missing_coverage_days=int(observed-n),unlisted_days_within_span=int(span-observed),first_date=str(g._date.min().date()),last_date=str(g._date.max().date()))
        for name,fn in CONDITIONS.items():
            condition=fn(g.coverage)&g.coverage.notna();r[name+'_day_fraction']=float(condition.sum()/n) if n else np.nan;r[name+'_longest_run_days']=longest_daily_run(g._date,condition)
        records.append(r)
    units=pd.DataFrame(records).rename(columns={'_unit':'unit'})
    aggregation={'mean_unit_coverage':('mean_coverage','mean'),'minimum_unit_coverage':('mean_coverage','min'),'units':('unit','size'),'units_with_coverage':('mean_coverage','count'),'valid_coverage_days':('valid_coverage_days','sum'),'missing_coverage_days':('missing_coverage_days','sum'),'unlisted_days_within_span':('unlisted_days_within_span','sum')}
    for name in CONDITIONS:aggregation['mean_unit_'+name+'_day_fraction']=(name+'_day_fraction','mean');aggregation['maximum_'+name+'_run_days']=(name+'_longest_run_days','max')
    if keys:summary=units.groupby(keys,as_index=False,dropna=False).agg(**aggregation)
    else:
        temp=units.assign(_all='all');summary=temp.groupby('_all',as_index=False).agg(**aggregation).drop(columns='_all')
    metadata=dict(unit_field=unit_col,date_field=date_col,group_fields=keys,k_filter=k,input_rows=len(data),selected_rows=len(z),fraction_denominator='每个体非缺失覆盖日；再按个体等权平均',missing_policy='缺日期或缺覆盖均打断连续段；跨度仅各个体首末记录内',thresholds=[0,.1,.25,.5],automatic_acceptance_decision=False)
    return units,summary,metadata

def write_report(units,summary,meta,out):
    out.mkdir(parents=True,exist_ok=True);units.to_csv(out/'Per unit audit.csv',index=False);summary.to_csv(out/'Group summary.csv',index=False);(out/'Audit metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    labels={'mean_unit_coverage':'个体等权平均覆盖','minimum_unit_coverage':'最低个体平均覆盖','unit':'个体/单元','mean_coverage':'个体平均覆盖','observed_days':'有行日期数','valid_coverage_days':'有效覆盖日期数','missing_coverage_days':'缺覆盖日期数','unlisted_days_within_span':'首末日期内未列日期数','units':'个体数','units_with_coverage':'有覆盖值个体数'}
    for name,title in [('zero','零覆盖'),('below10','低于10%'),('below25','低于25%'),('below50','低于50%')]:
        labels[name+'_day_fraction']=title+'日比例';labels[name+'_longest_run_days']=title+'最长连续日';labels['mean_unit_'+name+'_day_fraction']='个体等权'+title+'日比例';labels['maximum_'+name+'_run_days']='最大'+title+'连续日'
    tables=summary.rename(columns=labels).to_html(index=False,float_format=lambda x:f'{x:.4f}',escape=True)+units.rename(columns=labels).to_html(index=False,float_format=lambda x:f'{x:.4f}',escape=True)
    html='<!doctype html><html lang="zh"><meta charset="utf-8"><title>监测覆盖审计</title><style>body{font-family:sans-serif;margin:2em;line-height:1.6}table{border-collapse:collapse;font-size:13px}th,td{padding:5px;border:1px solid #ccc}th{background:#eef3f6}</style><h1>监测覆盖审计</h1><p>同时查看平均覆盖、最低个体覆盖和低覆盖持续时间。0、10%、25%、50%仅为描述阈值，本工具不自动判定达标，也不使用后期覆盖重新优化配置。</p><p>缺日期或缺覆盖打断连续段；比例分母为各个体有覆盖值的日期，再按个体等权平均。未列日期仅在各个体首末记录间计数，不能解释为动物不存在或已遗漏。</p><p>输入身份字段：'+escape(meta['unit_field'])+'；日期字段：'+escape(meta['date_field'])+'。</p><h2>各配置摘要</h2>'+tables.split('</table>',1)[0]+'</table><h2>各个体完整审计</h2>'+tables.split('</table>',1)[1]+'</html>'
    (out/'Coverage audit.html').write_text(html,encoding='utf-8')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('input',type=Path);p.add_argument('--out',required=True,type=Path);p.add_argument('--k',type=int);p.add_argument('--unit-col');p.add_argument('--date-col');a=p.parse_args();d=pd.read_parquet(a.input) if a.input.suffix.lower()=='.parquet' else pd.read_csv(a.input);u,s,m=audit(d,a.unit_col,a.date_col,a.k);write_report(u,s,m,a.out);print(a.out/'Coverage audit.html')
if __name__=='__main__':main()
