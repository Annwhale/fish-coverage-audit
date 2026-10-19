#  _____ _____ ____   ____
# |  ___|  ___|  _ \ / ___|
# | |_  | |_  | |_) | |
# |  _| |  _| |  _ <| |___
# |_|   |_|   |_| \_\____|
# FFRC
"""三个手写个体，检查缺日期/缺覆盖打断以及个体等权。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from coverage_audit_20261019 import audit,write_report
ROOT=Path(__file__).resolve().parent
x=pd.DataFrame([
 ('A','2026-01-01',0),('A','2026-01-02',.05),('A','2026-01-04',0),('A','2026-01-05',0),
 ('B','2026-01-01',0),('B','2026-01-02',np.nan),('B','2026-01-03',0),
 ('C','2026-01-01',.2),('C','2026-01-02',.3),('C','2026-01-03',.6),
],columns=['unit','date','coverage'])
u,s,meta=audit(x);a=u.set_index('unit')
assert a.loc['A','zero_longest_run_days']==2
assert a.loc['A','below10_longest_run_days']==2 # 缺失01-03打断，不是四日连续。
assert a.loc['A','unlisted_days_within_span']==1
assert a.loc['B','zero_longest_run_days']==1 # 缺覆盖01-02打断。
assert a.loc['B','missing_coverage_days']==1
assert a.loc['B','zero_day_fraction']==1 # 只在两个非缺失日中统计。
assert a.loc['C','below50_longest_run_days']==2
assert a.loc['C','below25_day_fraction']==1/3
assert abs(s.mean_unit_coverage.iloc[0]-(.0125+0+(1.1/3))/3)<1e-12
assert s.minimum_unit_coverage.iloc[0]==0
try:audit(pd.concat([x,x.iloc[:1]],ignore_index=True))
except ValueError:pass
else:raise AssertionError('重复个体日期须拒绝')
out=ROOT/'small verification';write_report(u,s,meta,out);x.to_csv(out/'Handwritten input.csv',index=False);(out/'Verification.json').write_text(json.dumps({'three_units':True,'date_gap_breaks_runs':True,'missing_coverage_breaks_runs':True,'fraction_denominator_excludes_missing':True,'equal_unit_weighting':True,'duplicate_dates_rejected':True},indent=2)+'\n');print('手写验证通过')
