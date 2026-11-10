"""Post-hoc error diagnostics and a pre-specified 50/50 forecast ensemble; no fitting."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.demand import ROOT
from harness import dayahead

MODELS = ['nbeats', 'lightgbm', 'lstm', 'tft', 'gam', 'tcn']
LABELS = dict(nbeats='N-BEATSx', lightgbm='LightGBM', lstm='LSTM', tft='TFT',
              gam='GAM', tcn='TCN', ensemble='等权集成')


def metrics(actual, forecast):
    error = forecast - actual
    return dict(n=len(actual), mape=float(np.mean(abs(error)/actual)*100),
                mae=float(np.mean(abs(error))), bias_mw=float(np.mean(error)))


def markdown(frame, digits=3):
    lines = ['| ' + ' | '.join(map(str, frame.columns)) + ' |',
             '|' + '---|' * len(frame.columns)]
    for row in frame.itertuples(index=False, name=None):
        lines.append('| ' + ' | '.join(f'{v:.{digits}f}' if isinstance(v, (float, np.floating)) else str(v)
                                      for v in row) + ' |')
    return lines


def main():
    base = ROOT/'results/demand'
    out = base/'error_analysis'
    out.mkdir(exist_ok=True)
    data_path = ROOT/'data/nem_weather_halfhourly_2010_2026.csv'
    df = dayahead.load_dataset(data_path)
    df = df.loc[dayahead.market_day(df.index).year >= 2015]
    days = dayahead.market_day(df.index)
    start = df.index-pd.Timedelta(minutes=30)
    actual = df.nsw_demand.to_numpy()
    periods = {'2015–2024': days.year < 2025, '2025–2026.10': days.year >= 2025,
               '全期': np.ones(len(df), dtype=bool)}
    groups = {
        'year': days.year.astype(str), 'month': days.to_period('M').astype(str),
        'calendar_month': days.month.astype(str).str.zfill(2),
        'slot': start.strftime('%H:%M'),
        'day_type': np.where(days.dayofweek < 5, '工作日（周一至周五）', '周末'),
        'temperature': pd.cut(df.syd_temp, [-np.inf, 10, 20, 30, np.inf], right=False,
                              labels=['<10°C', '10–20°C', '20–30°C', '≥30°C']).astype(str).to_numpy()}
    summary, grouped, complement, monthly_effects, daily_rows = [], [], [], [], []
    provenance = {'weights': {'nbeats': .5, 'lightgbm': .5}, 'fit_performed': False,
                  'data_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(), 'sources': {}}
    for strategy in ['frozen', 'periodic']:
        predictions = {}
        for name in MODELS:
            root = base/('formal_gpu' if name in ['tft', 'tcn'] else 'formal')
            path = root/'backtest'/name/f'{strategy}_predictions.csv.gz'
            f = pd.read_csv(path, index_col='timestamp', parse_dates=True)
            assert f.index.equals(df.index)
            np.testing.assert_allclose(f.actual, actual, rtol=0, atol=1e-8)
            assert np.isfinite(f.forecast).all()
            predictions[name] = f.forecast.to_numpy()
            provenance['sources'][f'{strategy}/{name}'] = dict(path=str(path.relative_to(ROOT)),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        predictions['ensemble'] = (predictions['nbeats']+predictions['lightgbm'])/2
        pd.DataFrame({'actual': actual, 'forecast': predictions['ensemble']}, index=df.index).to_csv(
            out/f'{strategy}_ensemble_predictions.csv.gz', index_label='timestamp')
        # The convexity bound detects mixing forecast rows or averaging scores instead of predictions.
        ape = {name: abs(pred-actual)/actual*100 for name, pred in predictions.items()}
        assert np.all(ape['ensemble'] <= (ape['nbeats']+ape['lightgbm'])/2+1e-9)
        for period, mask in periods.items():
            for name, pred in predictions.items():
                summary.append(dict(strategy=strategy, period=period, model=name, **metrics(actual[mask], pred[mask])))
                for dimension, values in groups.items():
                    # Year/month tables already encode their periods, so avoid duplicating them.
                    if dimension in ['year', 'month'] and period != '全期':
                        continue
                    work = pd.DataFrame({'group': np.asarray(values)[mask], 'actual': actual[mask],
                                         'forecast': pred[mask], 'ape': ape[name][mask]})
                    total = work.ape.sum()
                    for key, sub in work.groupby('group', sort=True):
                        grouped.append(dict(strategy=strategy, period=period, model=name, dimension=dimension,
                            group=key, **metrics(sub.actual.to_numpy(), sub.forecast.to_numpy()),
                            sample_share_pct=len(sub)/len(work)*100,
                            ape_sum_share_pct=sub.ape.sum()/total*100))
            e1 = predictions['nbeats'][mask]-actual[mask]
            e2 = predictions['lightgbm'][mask]-actual[mask]
            row = dict(strategy=strategy, period=period, signed_error_correlation=float(np.corrcoef(e1,e2)[0,1]),
                       opposite_sign_pct=float(np.mean(e1*e2 < 0)*100))
            # Paired 7-day circular block bootstrap: descriptive sensitivity to serial dependence.
            rng = np.random.default_rng(42)
            for name in ['nbeats', 'lightgbm']:
                delta = (ape['ensemble'][mask]-ape[name][mask]).reshape(-1,48).mean(axis=1)
                boot = []
                for _ in range(1000):
                    indices = (rng.integers(len(delta), size=(int(np.ceil(len(delta)/7)),1))+np.arange(7))%len(delta)
                    boot.append(delta[indices.ravel()[:len(delta)]].mean())
                row[f'delta_vs_{name}_pp'] = float(delta.mean())
                row[f'ci_low_vs_{name}'], row[f'ci_high_vs_{name}'] = map(float, np.quantile(boot,[.025,.975]))
            complement.append(row)
        daily = pd.DataFrame({name: errors.reshape(-1,48).mean(1) for name, errors in ape.items()},
                             index=days[::48])
        daily.index.name = 'market_day'
        daily.to_csv(out/f'{strategy}_daily_mape.csv')
        for date, row in daily.nlargest(10,'ensemble').iterrows():
            daily_rows.append(dict(strategy=strategy, market_day=str(date.date()), **row.to_dict()))
    scores = pd.DataFrame(summary)
    detail = pd.DataFrame(grouped)
    # Every grouping must partition the same evaluation rows and reproduce its aggregate MAPE.
    for (strategy, period, name, dimension), sub in detail.groupby(['strategy','period','model','dimension']):
        reference = scores[(scores.strategy==strategy)&(scores.period==period)&(scores.model==name)].iloc[0]
        assert sub.n.sum() == reference.n
        np.testing.assert_allclose(np.average(sub.mape, weights=sub.n), reference.mape, rtol=0, atol=1e-10)
    for filename, frame in [('summary',scores),('grouped_errors',detail),('complementarity',pd.DataFrame(complement)),
                             ('worst_ensemble_days',pd.DataFrame(daily_rows))]:
        frame.to_csv(out/f'{filename}.csv', index=False)
    for strategy in ['frozen','periodic']:
        for dimension in ['year','month']:
            wide = detail[(detail.strategy == strategy)&(detail.dimension == dimension)].pivot(
                index='group', columns='model', values='mape')[MODELS+['ensemble']]
            wide.to_csv(out/f'{strategy}_{dimension}_comparison.csv', index_label=dimension)
            if dimension == 'month':
                for period, mask in [('2015–2024',wide.index < '2025-01'),('2025–2026.10',wide.index >= '2025-01'),
                                     ('全期',np.ones(len(wide),dtype=bool))]:
                    sub = wide.loc[mask]
                    monthly_effects.append(dict(strategy=strategy, period=period, months=len(sub),
                        wins_vs_nbeats=int((sub.ensemble < sub.nbeats).sum()),
                        wins_vs_lightgbm=int((sub.ensemble < sub.lightgbm).sum()),
                        wins_vs_both=int((sub.ensemble < sub[['nbeats','lightgbm']].min(axis=1)).sum())))
    pd.DataFrame(monthly_effects).to_csv(out/'monthly_wins.csv',index=False)
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    lines = ['# 误差分析与等权集成', '',
        '本轮只分析已保存的半小时预测，不重训、不搜索权重。集成预测 = 0.5 × N-BEATSx + 0.5 × LightGBM；'
        '冻结版和定期重训版各自配对，使用完全相同的日期和真实值。所有 MAPE 单位为 %。', '',
        '这些历史结果已经参与方案探索，属于事后诊断；近期段为 2025 年至 2026 年 10 月。', '',
        '## 集成能否改善 MAPE', '']
    def score(name, period, strategy='periodic'):
        return scores[(scores.model==name)&(scores.period==period)&(scores.strategy==strategy)].mape.iloc[0]
    def group_rows(dimension, period='2025–2026.10'):
        return detail[(detail.strategy=='periodic')&(detail.period==period)&
                      (detail.model=='ensemble')&(detail.dimension==dimension)]
    peak = group_rows('slot').sort_values('mape').iloc[-1]
    temp = group_rows('temperature').set_index('group')
    day_type = group_rows('day_type').set_index('group')
    win = next(r for r in monthly_effects if r['strategy']=='periodic' and r['period']=='全期')
    recent_win = next(r for r in monthly_effects if r['strategy']=='periodic' and r['period']=='2025–2026.10')
    findings = ['## 主要发现', '',
        f"- **定期重训等权集成有效**：全期 MAPE {score('ensemble','全期'):.3f}%，"
        f"相对较好的单模型降低 {min(score('nbeats','全期'),score('lightgbm','全期'))-score('ensemble','全期'):.3f} 个百分点；"
        f"近期为 {score('ensemble','2025–2026.10'):.3f}%。",
        f"- **改善跨越多个时期**：全期 {win['months']} 个月中 {win['wins_vs_both']} 个月同时优于两个组成模型；"
        f"近期为 {recent_win['wins_vs_both']}/{recent_win['months']} 个月。",
        f"- **冻结版集成没有超过较好的组成模型**：全期 {score('ensemble','全期','frozen'):.3f}%，"
        f"LightGBM 为 {score('lightgbm','全期','frozen'):.3f}%。不能把定期重训的集成收益推广到冻结版。",
        f"- **近期最难的时段集中在中午**：集成在 {peak['group']} 开始的半小时 MAPE 达 {peak.mape:.3f}%，"
        f"MAE {peak.mae:.1f} MW，平均偏差 +{peak.bias_mw:.1f} MW，存在整体高估。",
        f"- **高温误差大，但并非总误差的主要来源**：近期 ≥30°C 时 MAPE {temp.loc['≥30°C','mape']:.3f}%，"
        f"仅 {int(temp.loc['≥30°C','n'])} 个半小时，贡献 {temp.loc['≥30°C','ape_sum_share_pct']:.2f}% 的绝对百分比误差总和；"
        f"20–30°C 组贡献 {temp.loc['20–30°C','ape_sum_share_pct']:.2f}%。",
        f"- **近期周末较难**：集成周末 MAPE {day_type.loc['周末','mape']:.3f}%，"
        f"周一至周五 {day_type.loc['工作日（周一至周五）','mape']:.3f}%。", '',
        '下一步优先检验中午负荷形状、天气/辐射相关特征及训练窗口是否改善这些误差；'
        '当前分组不能单独证明太阳能或某个天气变量是原因。多种子复核应保留当前固定权重，避免同时改变多个因素。', '']
    lines[6:6] = findings
    focus = scores[scores.model.isin(['nbeats','lightgbm','ensemble'])].pivot(
        index=['strategy','period'],columns='model',values='mape').reset_index()
    focus['相对两者较优者变化（百分点）'] = focus.ensemble-focus[['nbeats','lightgbm']].min(axis=1)
    focus = focus.rename(columns={'strategy':'策略','period':'区间','nbeats':'N-BEATSx','lightgbm':'LightGBM','ensemble':'等权集成'})
    lines += markdown(focus) + ['', '变化为负表示集成优于该区间内较好的单模型。', '', '## 定期重训版年度比较', '']
    annual = pd.read_csv(out/'periodic_year_comparison.csv')
    annual['year'] = annual.year.astype(str).replace({'2026':'2026（1—10月）'})
    lines += markdown(annual[['year','nbeats','lightgbm','ensemble']].rename(columns=LABELS))
    lines += ['', '## 月度改善是否持续', ''] + markdown(pd.DataFrame(monthly_effects).rename(columns={
        'strategy':'策略','period':'区间','months':'月数','wins_vs_nbeats':'优于 N-BEATSx 月数',
        'wins_vs_lightgbm':'优于 LightGBM 月数','wins_vs_both':'同时优于两者月数'}))
    lines += ['', '## 误差集中位置：定期重训版', '',
        '下表覆盖六模型及集成。分组按市场日与区间开始时间计算；“工作日”只区分星期，不扣除公共假期。'
        '温度为目标半小时 Sydney 气温。分组比较是描述性关联，不能证明气温或星期导致误差。']
    for period in ['全期','2025–2026.10']:
        lines += ['', f'### {period}', '']
        for dimension, title in [('calendar_month','日历月份'),('day_type','工作日 / 周末'),('temperature','气温区间')]:
            sub = detail[(detail.strategy=='periodic')&(detail.period==period)&(detail.dimension==dimension)]
            wide = sub.pivot(index='group',columns='model',values='mape')[MODELS+['ensemble']]
            if dimension == 'temperature':
                wide = wide.reindex(['<10°C','10–20°C','20–30°C','≥30°C'])
            counts = sub[sub.model=='ensemble'].set_index('group').n
            wide.insert(0,'半小时数',counts)
            lines += ['', f'**{title}**', ''] + markdown(wide.rename(columns=LABELS).reset_index())
        slots = detail[(detail.strategy=='periodic')&(detail.period==period)&(detail.dimension=='slot')]
        worst = slots.sort_values('mape',ascending=False).groupby('model').head(1)
        lines += ['', '**每个模型误差最高的半小时时段（区间开始时间）**', ''] + markdown(
            worst[['model','group','n','mape','mae','bias_mw']].replace({'model':LABELS}).rename(columns={
                'model':'模型','group':'区间开始','n':'半小时数','mape':'MAPE (%)','mae':'MAE (MW)','bias_mw':'平均偏差 (MW)'}))
    lines += ['', '## 互补性与局限', '',
        'complementarity.csv 包含有符号 MW 误差相关系数、误差方向相反的比例，以及集成相对单模型的 MAPE 差值。'
        '差值区间使用配对 7 日循环区块 bootstrap，1,000 次抽样；未消除跨年非平稳性或事后选择偏差，只作描述性参考。', '',
        'grouped_errors.csv 同时记录样本数、样本占比、MAPE、MAE、平均有符号误差（预测−实际）、'
        '绝对百分比误差总和占比。高 MAPE 的小样本组不一定贡献最多总误差。', '',
        '完整记录：[分组误差](grouped_errors.csv) · [月度比较](periodic_month_comparison.csv) · '
        '[互补性](complementarity.csv) · [集成误差最大的日期](worst_ensemble_days.csv) · '
        '[来源校验和](provenance.json) · [原六模型结果](../formal_report/README.md)', '',
        '后续若根据诊断新增特征或修改模型超参数，仍在 Stage 1 允许的数据中选择；'
        '适应和集成策略的历史比较按探索性结果报告。本轮集成不是最终部署模型。']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    print(focus.to_string(index=False))
    print(pd.DataFrame(monthly_effects).to_string(index=False))


if __name__ == '__main__':
    main()
