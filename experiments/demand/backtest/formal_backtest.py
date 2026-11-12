"""Frozen and common periodic backtests for Stage 1 selected configurations."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.demand import tabular as demand_tabular
from experiments.demand import ROOT
from harness import dayahead


def neural_refit(name,config,epochs,df,start,end,eval_end,folder,device="cpu",seed=42):
    import torch
    from torch.nn import functional as F
    from src.demand.neural import DailyWindows,make_model,predict,device_batch,train_loss
    torch.set_num_threads(2)
    data=DailyWindows(df,config,start,end,eval_end)
    model=make_model(name,config,seed).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=config["lr"])
    path=folder/"trained.pt"
    if path.exists():
        state=torch.load(path,weights_only=False,map_location=device)
        if state["config"]!=config or state["epochs"]!=epochs or state.get("seed",42)!=seed:
            raise ValueError("Refit checkpoint config mismatch")
        model.load_state_dict(state["model"])
    else:
        for epoch in range(epochs):
            model.train()
            for batch in data.train_indices[torch.randperm(len(data.train_indices))].split(32):
                optimizer.zero_grad()
                loss=train_loss(data,model(device_batch(data,batch,device)).squeeze(-1),batch,device)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite refit loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1)
                optimizer.step()
        temp=folder/"trained.tmp"
        torch.save(dict(config=config,epochs=epochs,seed=seed,model=model.state_dict(),
                        demand_mean=data.mean,demand_scale=data.scale,
                        temp_mean=data.temp_mean,temp_scale=data.temp_scale),temp)
        temp.replace(path)
    prediction=predict(model,data,data.eval_indices).ravel()
    mask=np.repeat(np.isin(np.arange(len(data.target)),data.eval_indices.numpy()),48)
    return pd.Series(prediction,index=data.index[mask])


def run_model(name,df,root):
    selected=json.loads((root/name/"selected.json").read_text())
    config=selected["config"]
    base=root/"backtest"/name
    base.mkdir(parents=True,exist_ok=True)
    selected_path=base/"selection.json"
    if selected_path.exists() and json.loads(selected_path.read_text())!=selected:
        raise ValueError("Backtest exists for a different selection. Choose a new output root.")
    selected_path.write_text(json.dumps(selected,indent=2))
    days=dayahead.market_day(df.index)
    tabular=name in ("gam","lightgbm")
    if tabular:
        x=demand_tabular.add_holidays(demand_tabular.features(df,selected["variant"]),
                                      selected.get("holiday_level",0))
    periodic=[]
    # frozen_only: the single 2010-2014 fit (Stage 2 requires frozen results for every model).
    for year in ([2015] if selected.get("frozen_only") else range(2015,2026,2)):
        folder=base/str(year)
        folder.mkdir(exist_ok=True)
        path=folder/"predictions.csv.gz"
        # train_years: periodic window length in years (default 4) or "all" (expanding from 2010).
        years=selected.get("train_years",4)
        start=pd.Timestamp("2010-01-01" if year==2015 or years=="all" else f"{year-years}-01-01")
        end=pd.Timestamp(f"{year-1}-12-31")
        eval_end=pd.Timestamp("2026-10-31" if year in (2015,2025) else f"{year+1}-12-31")
        if path.exists():
            frame=pd.read_csv(path,index_col="timestamp",parse_dates=True)
        else:
            begin=time.monotonic()
            if tabular:
                train=(days>=start)&(days<=end)
                test=(days>end)&(days<=eval_end)
                pred=demand_tabular.forecast(name,x,df.nsw_demand,train,test,config)
                pred=pd.Series(pred,index=df.index[test])
            else:
                pred=neural_refit(name,config,selected.get("refit_epochs_by_year",{}).get(str(year),selected["refit_epochs"]),df,start,end,eval_end,folder,
                                  selected.get("device","cpu"),selected.get("refit_seed",42))
            frame=pd.DataFrame(dict(actual=df.nsw_demand.loc[pred.index],forecast=pred))
            frame.index.name="timestamp"
            if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
                raise ValueError("Invalid backtest output")
            frame.to_csv(path)
            (folder/"fit.json").write_text(json.dumps(dict(train_start=str(start),train_end=str(end),
                 eval_end=str(eval_end),seed=selected.get("refit_seed",42),elapsed_seconds=time.monotonic()-begin),indent=2))
            print(name,"refit",year,"finished",flush=True)
        if year==2015:
            frame.to_csv(base/"frozen_predictions.csv.gz")
        segment=frame[dayahead.market_day(frame.index).year<year+2]
        periodic.append(segment)
    if selected.get("frozen_only"):
        print(name,"frozen backtest complete",flush=True)
        return
    combined=pd.concat(periodic)
    expected=df.index[days.year>=2015]
    if not combined.index.equals(expected):
        raise ValueError("Missing or duplicated forecast dates")
    combined.to_csv(base/"periodic_predictions.csv.gz")
    print(name,"backtest complete",flush=True)


def summarize(root, sources=None):
    records=[]
    monthly=[]
    annual=[]
    for name in ("gam","lightgbm","nbeats","tft","tcn","lstm"):
        for strategy in ("frozen","periodic"):
            source = sources.get(name, root) if sources else root
            path=source/"backtest"/name/f"{strategy}_predictions.csv.gz"
            if not path.exists():
                continue
            frame=pd.read_csv(path,index_col="timestamp",parse_dates=True)
            days=dayahead.market_day(frame.index)
            for label,mask in [("2015–2024",days.year<2025),("2025–2026.10",days.year>=2025),
                               ("overall",days.year>=2015)]:
                sub=frame.loc[mask]
                records.append(dict(model=name,strategy=strategy,period=label,n=len(sub),
                                    mape=dayahead.mape(sub.actual,sub.forecast),mae=dayahead.mae(sub.actual,sub.forecast)))
            for period,group in frame.groupby(days.year):
                annual.append(dict(model=name,strategy=strategy,year=period,n=len(group),
                                   mape=dayahead.mape(group.actual,group.forecast),mae=dayahead.mae(group.actual,group.forecast)))
            for period,group in frame.groupby(days.to_period("M")):
                ape=100*abs(group.actual-group.forecast)/group.actual
                means=ape.to_numpy().reshape(-1,48).mean(1)
                bootstrap=np.random.default_rng(42).choice(means,size=(1000,len(means))).mean(1)
                low,high=np.quantile(bootstrap,[.025,.975])
                margin=1.96*ape.std()/np.sqrt(len(ape))
                monthly.append(dict(model=name,strategy=strategy,month=str(period),n=len(group),
                                    mape=ape.mean(),mae=dayahead.mae(group.actual,group.forecast),
                                    iid_low=ape.mean()-margin,iid_high=ape.mean()+margin,
                                    bootstrap_low=low,bootstrap_high=high))
    for label,data in [("comparison",records),("monthly",monthly),("annual",annual)]:
        pd.DataFrame(data).to_csv(root/f"{label}.csv",index=False)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models",nargs="+",default=["gam","lightgbm","nbeats","tft","tcn","lstm"])
    parser.add_argument("--root",type=Path,default=ROOT/"results/demand/formal")
    args=parser.parse_args()
    table_models=[name for name in args.models if name in ("gam","lightgbm")]
    neural_models=[name for name in args.models if name not in ("gam","lightgbm")]
    if table_models and neural_models:
        for group in (table_models,neural_models):
            subprocess.run([sys.executable,"-m","experiments.demand.backtest.formal_backtest",
                            "--root",str(args.root),"--models",*group],check=True)
        sys.exit(0)
    df=dayahead.load_dataset(ROOT/"data/nem_weather_halfhourly_2010_2026.csv")
    for name in args.models:
        run_model(name,df,args.root)
    summarize(args.root)
