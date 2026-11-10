"""Verify whether changing only LightGBM's seed changes any formal backtest fit."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.demand.tabular import features,forecast
from harness import dayahead
root=Path('results/demand/formal')
Path('results/demand/lightgbm_seed_check').mkdir(parents=True,exist_ok=True)
s=json.loads((root/'lightgbm/selected.json').read_text())
df=dayahead.load_dataset('data/nem_weather_halfhourly_2010_2026.csv'); days=dayahead.market_day(df.index); x=features(df,s['variant'])
rows=[]
for year in range(2015,2026,2):
 start=pd.Timestamp('2010-01-01' if year==2015 else f'{year-4}-01-01'); end=pd.Timestamp(f'{year-1}-12-31')
 stop=pd.Timestamp('2026-10-31' if year in (2015,2025) else f'{year+1}-12-31')
 train=(days>=start)&(days<=end); test=(days>end)&(days<=stop)
 old=pd.read_csv(root/'backtest/lightgbm'/str(year)/'predictions.csv.gz',index_col='timestamp',parse_dates=True)
 assert old.index.equals(df.index[test])
 for seed in [43,44]:
  pred=forecast('lightgbm',x,df.nsw_demand,train,test,dict(s['config'],random_state=seed))
  delta=float(np.max(abs(pred-old.forecast.to_numpy())))
  rows.append(dict(refit_year=year,seed=seed,n=len(pred),max_abs_difference_from_seed42_mw=delta,mape=dayahead.mape(old.actual,pred)))
  print(year,seed,delta,flush=True)
pd.DataFrame(rows).to_csv('results/demand/lightgbm_seed_check/comparison.csv',index=False)
assert max(r['max_abs_difference_from_seed42_mw'] for r in rows)<1e-9
print('All six fit windows: seeds 43 and 44 reproduce seed 42 predictions within 1e-9 MW.')
