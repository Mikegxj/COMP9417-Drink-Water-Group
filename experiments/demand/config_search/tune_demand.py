"""Bounded Stage 1 search and restored-checkpoint seed replication."""
import argparse
import itertools
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

MODELS = ["gam","lightgbm","nbeats","tft","tcn","lstm"]
VARIANTS = {"gam":"prior_extrema","lightgbm":"cloud"}


def configurations(name):
    if name=="gam":
        return [dict(n_knots=k,alpha=a) for k,a in itertools.product([3,5,7,9],[0.1,1,10])]
    if name=="lightgbm":
        return [dict(num_leaves=l,min_child_samples=n,learning_rate=r,n_estimators=500)
                for l,n,r in itertools.product([15,31,63],[24,48],[.02,.04])]
    if name=="nbeats":
        return [dict(window=w,width=128,weather=e,lr=r) for w,e,r in
                itertools.product([336,480],[False,True],[.001,.002])]
    return [dict(window=w,width=d,weather=True,lr=r)
            for w,d,r in itertools.product([336,480],[16,32],[.001,.002])]


def neural_trial(name, config, data, seed, folder, limit, device="cpu"):
    import torch
    from torch.nn import functional as F
    from src.demand.neural import make_model, predict, device_batch, train_loss
    torch.set_num_threads(2)
    folder.mkdir(parents=True,exist_ok=True)
    model = make_model(name,config,seed).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=config["lr"])
    best, best_epoch, first, elapsed = float("inf"),0,0,0.
    curves = []
    path=folder/"last.pt"
    if path.exists():
        saved=torch.load(path,weights_only=False,map_location="cpu")
        if saved["config"]!=config or saved["seed"]!=seed or saved.get("device","cpu")!=device:
            raise ValueError(f"Resume mismatch {folder}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.set_rng_state(saved["rng"])
        if device=="cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        best,best_epoch,first,curves,elapsed=saved["best"],saved["best_epoch"],saved["epoch"],saved["curves"],saved["elapsed"]
    start=time.monotonic()
    for epoch in range(first+1,limit+1):
        if epoch-1-best_epoch>=20:
            break
        model.train()
        train_error=0.
        for batch in data.train_indices[torch.randperm(len(data.train_indices))].split(32):
            optimizer.zero_grad()
            output=model(device_batch(data,batch,device)).squeeze(-1)
            loss=train_loss(data,output,batch,device)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss in {folder}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            train_error+=loss.item()*len(batch)
        # Daily forecast horizon and inverse-scale MAPE, including all seasons.
        prediction=predict(model,data,data.eval_indices)
        actual=data.actual[data.eval_indices]
        error=dayahead.mape(actual,prediction)
        curves.append(dict(epoch=epoch,train_mae=train_error/len(data.train_indices)*data.scale,
                           val_mape=error,val_mae=dayahead.mae(actual,prediction)))
        if error<best:
            best,best_epoch=error,epoch
            torch.save(model.state_dict(),folder/"best.pt")
        state=dict(config=config,seed=seed,device=device,
                   cuda_rng=torch.cuda.get_rng_state_all() if device=="cuda" else None,model=model.state_dict(),optimizer=optimizer.state_dict(),
                   rng=torch.get_rng_state(),best=best,best_epoch=best_epoch,epoch=epoch,
                   curves=curves,elapsed=elapsed+time.monotonic()-start)
        temporary=folder/"last.tmp"
        torch.save(state,temporary)
        temporary.replace(path)
        pd.DataFrame(curves).to_csv(folder/"curve.csv",index=False)
    model.load_state_dict(torch.load(folder/"best.pt",weights_only=True))
    prediction=predict(model,data,data.eval_indices)
    actual=data.actual[data.eval_indices]
    mape=dayahead.mape(actual,prediction)
    if abs(mape-best)>1e-4:
        raise ValueError("Checkpoint did not reproduce validation score")
    index=data.index
    mask=np.repeat(np.isin(np.arange(len(data.target)),data.eval_indices.numpy()),48)
    frame=pd.DataFrame(dict(actual=actual.ravel(),forecast=prediction.ravel()),index=index[mask])
    frame.index.name="timestamp"
    frame.to_csv(folder/"predictions.csv.gz")
    report=dict(config=config,seed=seed,device=device,mape=mape,mae=dayahead.mae(actual,prediction),
                best_epoch=best_epoch,epochs_run=curves[-1]["epoch"],
                stop_reason="patience" if curves[-1]["epoch"]-best_epoch>=20 else "budget",
                elapsed_seconds=elapsed+time.monotonic()-start)
    (folder/"summary.json").write_text(json.dumps(report,indent=2))
    return report


def tune(name,df,folder,device="cpu"):
    folder.mkdir(parents=True,exist_ok=True)
    configs=configurations(name)
    if name in VARIANTS:
        days=dayahead.market_day(df.index)
        x=demand_tabular.features(df,VARIANTS[name])
        scores=[]
        for i,config in enumerate(configs):
            trial=folder/f"config_{i}"
            trial.mkdir(exist_ok=True)
            path=trial/"summary.json"
            if path.exists():
                report=json.loads(path.read_text())
            else:
                start=time.monotonic()
                pred=demand_tabular.forecast(name,x,df.nsw_demand,days.year<2014,days.year==2014,config)
                actual=df.nsw_demand[days.year==2014]
                report=dict(config=config,mape=dayahead.mape(actual,pred),mae=dayahead.mae(actual,pred),
                            elapsed_seconds=time.monotonic()-start)
                pd.DataFrame(dict(actual=actual,forecast=pred)).to_csv(trial/"predictions.csv.gz")
                path.write_text(json.dumps(report,indent=2))
            scores.append(dict(config_id=i,**report))
            print(name,i,round(report["mape"],4),flush=True)
        best=min(scores,key=lambda row:row["mape"])
        selected=dict(model=name,variant=VARIANTS[name],**best)
    else:
        from src.demand.neural import DailyWindows
        data_cache={}
        def get_data(config):
            key=(config["window"],config["weather"])
            if key not in data_cache:
                data_cache[key]=DailyWindows(df,config,pd.Timestamp("2010-01-01"),
                                            pd.Timestamp("2013-12-31"),pd.Timestamp("2014-12-31"))
            return data_cache[key]
        scores=[]
        # Same 32-epoch initial resource budget across all neural configurations.
        for i,config in enumerate(configs):
            report=neural_trial(name,config,get_data(config),42,folder/f"config_{i}/seed_42",32,device)
            scores.append(dict(config_id=i,**report))
            print(name,"screen",i,round(report["mape"],4),flush=True)
        finalists=sorted(scores,key=lambda r:r["mape"])[:2]
        repeated=[]
        for row in finalists:
            i,config=row["config_id"],row["config"]
            for seed in (42,43,44):
                report=neural_trial(name,config,get_data(config),seed,folder/f"config_{i}/seed_{seed}",160,device)
                if report["stop_reason"]=="budget":
                    report=neural_trial(name,config,get_data(config),seed,folder/f"config_{i}/seed_{seed}",320,device)
                repeated.append(dict(config_id=i,**report))
                print(name,"finalist",i,seed,round(report["mape"],4),report["best_epoch"],report["stop_reason"],flush=True)
        stats=pd.DataFrame(repeated).groupby("config_id").agg(mape=("mape","mean"),std=("mape","std"))
        best_id=int(stats.mape.idxmin())
        records=[r for r in repeated if r["config_id"]==best_id]
        selected=dict(model=name,device=device,config_id=best_id,config=configs[best_id],
                      mape=float(stats.loc[best_id,"mape"]),std=float(stats.loc[best_id,"std"]),
                      refit_epochs=int(np.median([r["best_epoch"] for r in records])),
                      seeds=[42,43,44],implementation=("src.demand.torch_models (matches neuralforecast 3.3.0)" if name in ("nbeats","tft") else "neuralforecast 3.3.0"),
                      any_budget_stop=any(r["stop_reason"]=="budget" for r in records))
        pd.DataFrame(repeated).to_csv(folder/"finalists.csv",index=False)
    pd.DataFrame(scores).to_csv(folder/"search.csv",index=False)
    (folder/"selected.json").write_text(json.dumps(selected,indent=2))
    return selected


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models",nargs="+",choices=MODELS,default=MODELS)
    parser.add_argument("--output",type=Path,default=ROOT/"results/demand/formal")
    parser.add_argument("--device",choices=["cpu","cuda"],default="cpu")
    args=parser.parse_args()
    table_models=[name for name in args.models if name in VARIANTS]
    neural_models=[name for name in args.models if name not in VARIANTS]
    if table_models and neural_models:
        for group in (table_models,neural_models):
            subprocess.run([sys.executable,"-m","experiments.demand.config_search.tune_demand",
                            "--output",str(args.output),"--device",args.device,"--models",*group],check=True)
        sys.exit(0)
    data=dayahead.load_dataset(ROOT/"data/nem_weather_halfhourly_2010_2026.csv")
    data=data.loc[dayahead.market_day(data.index).year<=2014]
    for name in args.models:
        tune(name,data,args.output/name,args.device)
