"""Fixed-configuration seed sensitivity; no retuning, ensemble or seed selection."""
import hashlib
import json
import platform
from importlib.metadata import version

import numpy as np
import pandas as pd

from experiments.demand.backtest.formal_backtest import run_model
from experiments.demand import ROOT
from harness import dayahead


def main():
    base=ROOT/'results/demand'
    out=base/'nbeats_seed_repeats'
    out.mkdir(exist_ok=True)
    original=json.loads((base/'formal/nbeats/selected.json').read_text())
    dataset=ROOT/'data/nem_weather_halfhourly_2010_2026.csv'
    df=dayahead.load_dataset(dataset)
    protocol=dict(seeds=[42,43,44],config=original['config'],refit_epochs=original['refit_epochs'],
        device='cpu',seed_42_source='results/demand/formal/backtest/nbeats',
        initial_training='2010–2014',refit_years=[2017,2019,2021,2023,2025],window_years=4,
        python=platform.python_version(),versions={p:version(p) for p in ['torch','neuralforecast','numpy']},
        data_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest())
    original_protocol=json.loads((base/'formal/protocol.json').read_text())
    assert protocol['data_sha256']==original_protocol['data_sha256']
    assert protocol['python']==original_protocol['python']
    for package, installed in protocol['versions'].items():
        assert installed==original_protocol['versions'][package], f'Environment changed: {package}'
    path=out/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError('Repeat protocol changed; use a new directory')
    path.write_text(json.dumps(protocol,indent=2))
    for seed in [43,44]:
        root=out/f'seed_{seed}'
        folder=root/'nbeats'
        folder.mkdir(parents=True,exist_ok=True)
        selected=dict(original,refit_seed=seed,device='cpu')
        path=folder/'selected.json'
        if path.exists() and json.loads(path.read_text())!=selected:
            raise ValueError('Selection changed')
        path.write_text(json.dumps(selected,indent=2))
        print(f'Starting seed {seed}',flush=True)
        run_model('nbeats',df,root)
    expected=df.loc[dayahead.market_day(df.index).year>=2015,'nsw_demand']
    days=dayahead.market_day(expected.index)
    summaries,annual,monthly,checksums=[],[],[],{}
    for seed in [42,43,44]:
        root=base/'formal' if seed==42 else out/f'seed_{seed}'
        for strategy in ['frozen','periodic']:
            path=root/'backtest/nbeats'/f'{strategy}_predictions.csv.gz'
            frame=pd.read_csv(path,index_col='timestamp',parse_dates=True)
            assert frame.index.equals(expected.index)
            np.testing.assert_allclose(frame.actual,expected,rtol=0,atol=1e-8)
            assert np.isfinite(frame.to_numpy()).all()
            checksums[f'{seed}/{strategy}']=hashlib.sha256(path.read_bytes()).hexdigest()
            if seed!=42:
                import torch
                for year in range(2015,2026,2):
                    state=torch.load(root/'backtest/nbeats'/str(year)/'trained.pt',weights_only=False,map_location='cpu')
                    assert state['seed']==seed and state['epochs']==original['refit_epochs']
                    assert state['config']==original['config']
                    fit=json.loads((root/'backtest/nbeats'/str(year)/'fit.json').read_text())
                    assert pd.Timestamp(fit['train_end'])==pd.Timestamp(f'{year-1}-12-31')
                    assert pd.Timestamp(fit['train_start'])==pd.Timestamp(f'{2010 if year==2015 else year-4}-01-01')
            for period,mask in [('2015–2024',days.year<2025),('2025–2026.10',days.year>=2025),('全期',days.year>=2015)]:
                sub=frame.loc[mask]
                summaries.append(dict(seed=seed,strategy=strategy,period=period,n=len(sub),
                    mape=dayahead.mape(sub.actual,sub.forecast),mae=dayahead.mae(sub.actual,sub.forecast)))
            for dimension,groups,records in [('year',days.year,annual),('month',days.to_period('M'),monthly)]:
                for label,sub in frame.groupby(groups):
                    records.append(dict(seed=seed,strategy=strategy,**{dimension:str(label)},n=len(sub),
                        mape=dayahead.mape(sub.actual,sub.forecast),mae=dayahead.mae(sub.actual,sub.forecast)))
        frozen=pd.read_csv(root/'backtest/nbeats/frozen_predictions.csv.gz')
        periodic=pd.read_csv(root/'backtest/nbeats/periodic_predictions.csv.gz')
        np.testing.assert_allclose(frozen.forecast[days.year<2017],periodic.forecast[days.year<2017],rtol=0,atol=1e-8)
    summary=pd.DataFrame(summaries)
    assert len(summary)==18 and len(annual)==72 and len(monthly)==852
    stats=summary.groupby(['strategy','period']).agg(mean_mape=('mape','mean'),std_mape=('mape','std'),
                                                   min_mape=('mape','min'),max_mape=('mape','max')).reset_index()
    for name,frame in [('summary',summary),('seed_statistics',stats),('annual',pd.DataFrame(annual)),('monthly',pd.DataFrame(monthly))]:
        frame.to_csv(out/f'{name}.csv',index=False)
    (out/'prediction_hashes.json').write_text(json.dumps(checksums,indent=2))
    (out/'source_hashes.json').write_text(json.dumps({name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        for name in ['experiments/demand/backtest/repeat_nbeats_backtest.py','experiments/demand/backtest/formal_backtest.py',
                     'src/demand/neural.py']},indent=2))
    def table(frame):
        result=['| '+' | '.join(map(str,frame.columns))+' |','|'+'---|'*len(frame.columns)]
        for row in frame.itertuples(index=False,name=None):
            result.append('| '+' | '.join(f'{v:.3f}' if isinstance(v,float) else str(v) for v in row)+' |')
        return result
    lines=['# N-BEATSx 三种子跨年稳定性复核','',
        '固定已选配置：10 天历史、宽度 128、天气/日历输入、学习率 0.001、59 轮。'
        '保持原训练窗口和五次重训日期，每次重新初始化使用同一实验种子。种子 42 复用原预测，43、44 在相同本机 CPU 环境补跑。', '',
        '本轮报告所有预先指定种子，不按历史成绩挑选种子，也不做预测集成。MAPE 单位为 %；'
        '均值是三个独立运行的 MAPE 均值，不是平均预测的 MAPE。'
        '种子标准差为三个结果的样本标准差，单位为百分点，不是未来预测误差的置信区间。','',
        '## 分期结果','']
    wide=summary.pivot(index=['strategy','period'],columns='seed',values='mape').reset_index()
    merged=wide.merge(stats,on=['strategy','period']).rename(columns={'strategy':'策略','period':'区间',
        42:'种子 42',43:'种子 43',44:'种子 44','mean_mape':'均值','std_mape':'标准差','min_mape':'最小值','max_mape':'最大值'})
    lines+=table(merged)
    reference=pd.read_csv(base/'formal_report/comparison.csv')
    comparisons=[]
    for period in ['2015–2024','2025–2026.10','全期']:
        label='overall' if period=='全期' else period
        lgb=float(reference[(reference.model=='lightgbm')&(reference.strategy=='periodic')&
                            (reference.period==label)].mape.iloc[0])
        sub=summary[(summary.strategy=='periodic')&(summary.period==period)]
        comparisons.append({'区间':period,'N-BEATSx 均值':sub.mape.mean(),'N-BEATSx 标准差':sub.mape.std(),
                            'LightGBM 现有结果':lgb,'低于 LightGBM 的种子数':int((sub.mape<lgb).sum())})
    lines+=['','## 与现有 LightGBM 定期重训结果对照','']+table(pd.DataFrame(comparisons))
    lines+=['','LightGBM 保持既有配置；补充检查在全部六个拟合窗口重跑种子 43、44，'
            '与原种子 42 的逐点预测最大差异约 1.82e-12 MW（CSV 浮点读写精度量级）。'
            '因此本配置下更换这些种子没有可测的预测变化。'
            '[LightGBM 种子检查](../lightgbm_seed_check/comparison.csv)。此表不是统计显著性检验。'
            '后续新方案应使用同一组种子进行配对比较，避免将随机波动误认为稳定收益。']
    annual_df=pd.DataFrame(annual)
    for strategy,title in [('frozen','冻结版'),('periodic','定期重训版')]:
        wide=annual_df[annual_df.strategy==strategy].pivot(index='year',columns='seed',values='mape')
        wide['均值']=wide[[42,43,44]].mean(axis=1)
        wide['标准差']=wide[[42,43,44]].std(axis=1)
        wide=wide.reset_index().rename(columns={'year':'年份',42:'种子 42',43:'种子 43',44:'种子 44'})
        wide['年份']=wide['年份'].replace({'2026':'2026（1—10月）'})
        lines+=['',f'## {title}年度 MAPE','']+table(wide)
    lines+=['','全期为 2015 年 1 月至 2026 年 10 月；近期段为 2025 年至 2026 年 10 月，已经用于探索。'
        '本轮保持其他条件一致，测量初始化与训练随机顺序带来的波动，不能代表不同数据抽样、架构或超参数的不确定性。','',
        '[逐月结果](monthly.csv) · [每次运行结果](summary.csv) · [执行协议](protocol.json) · '
        '[预测文件校验和](prediction_hashes.json) · [正式实验结果](../formal_report/README.md)']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(merged.to_string(index=False),flush=True)


if __name__=='__main__':
    main()
