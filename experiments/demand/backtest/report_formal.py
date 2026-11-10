"""Validate and consolidate all six completed runs without mixing CPU/GPU trials."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.demand.backtest.formal_backtest import summarize
from experiments.demand import ROOT
from harness import dayahead

MODELS = ("gam", "lightgbm", "nbeats", "tft", "tcn", "lstm")


def main():
    base = ROOT / "results/demand"
    output = base / "formal_report"
    sources = {name: base / ("formal_gpu" if name in ("tft", "tcn") else "formal")
               for name in MODELS}
    data_path = ROOT / "data/nem_weather_halfhourly_2010_2026.csv"
    data_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
    assert json.loads((base / "formal/protocol.json").read_text())["data_sha256"] == data_hash
    assert json.loads((base / "formal_gpu/execution.json").read_text())["data_sha256"] == data_hash
    df = dayahead.load_dataset(data_path)
    expected = df.loc[dayahead.market_day(df.index).year >= 2015, "nsw_demand"]
    manifest, selection = {}, []
    for name, source in sources.items():
        selected = json.loads((source / name / "selected.json").read_text())
        backtest = source / "backtest" / name
        assert json.loads((backtest / "selection.json").read_text()) == selected, name
        frames, checksums = {}, {}
        for strategy in ("frozen", "periodic"):
            path = backtest / f"{strategy}_predictions.csv.gz"
            frame = pd.read_csv(path, index_col="timestamp", parse_dates=True)
            assert frame.index.equals(expected.index), (name, strategy, "dates")
            assert np.isfinite(frame.to_numpy()).all(), (name, strategy, "finite")
            np.testing.assert_allclose(frame.actual, expected, rtol=0, atol=1e-8)
            frames[strategy] = frame
            checksums[strategy] = hashlib.sha256(path.read_bytes()).hexdigest()
        first = dayahead.market_day(expected.index).year < 2017
        np.testing.assert_allclose(frames["frozen"].forecast[first],
                                   frames["periodic"].forecast[first], rtol=0, atol=1e-8)
        for year in range(2015, 2026, 2):
            fit = json.loads((backtest / str(year) / "fit.json").read_text())
            assert pd.Timestamp(fit["train_end"]) == pd.Timestamp(f"{year-1}-12-31")
            start = 2010 if year == 2015 else year-4
            assert pd.Timestamp(fit["train_start"]) == pd.Timestamp(f"{start}-01-01")
        manifest[name] = {"source": str(source.relative_to(ROOT)), "sha256": checksums,
                          "selected": selected, "forecast_rows_per_strategy": len(expected),
                          "forecast_ranges": {strategy: [float(frame.forecast.min()),
                                                          float(frame.forecast.max())]
                                              for strategy, frame in frames.items()}}
        selection.append(dict(model=name, validation_mape=selected["mape"],
                              seed_std=selected.get("std"), device=selected.get("device", "cpu"),
                              refit_epochs=selected.get("refit_epochs"),
                              budget_stop=selected.get("any_budget_stop", False),
                              config=json.dumps(selected["config"])))
    output.mkdir(parents=True, exist_ok=True)
    summarize(output, sources)
    baseline_rows, baseline_annual = [], []
    days = dayahead.market_day(expected.index)
    for lag in (1, 7):
        forecast = df.nsw_demand.shift(48*lag).loc[expected.index]
        for day in ("2015-01-01", "2020-02-29", "2026-10-31"):
            history, weather = dayahead.make_inputs(df, day)
            official = dayahead.baseline(history, weather, lag_days=lag).nsw_demand_mw
            np.testing.assert_allclose(forecast.loc[official.index], official)
        for period, mask in (("2015–2024", days.year < 2025),
                             ("2025–2026.10", days.year >= 2025), ("overall", days.year >= 2015)):
            baseline_rows.append(dict(lag_days=lag, period=period, n=int(mask.sum()),
                                      mape=dayahead.mape(expected[mask], forecast[mask]),
                                      mae=dayahead.mae(expected[mask], forecast[mask])))
        for year in sorted(days.year.unique()):
            mask = days.year == year
            baseline_annual.append(dict(lag_days=lag, year=int(year), n=int(mask.sum()),
                                        mape=dayahead.mape(expected[mask], forecast[mask]),
                                        mae=dayahead.mae(expected[mask], forecast[mask])))
    pd.DataFrame(baseline_rows).to_csv(output / "baselines.csv", index=False)
    pd.DataFrame(baseline_annual).to_csv(output / "baselines_annual.csv", index=False)
    pd.DataFrame(selection).to_csv(output / "selection.csv", index=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    comparison = pd.read_csv(output / "comparison.csv")
    assert len(comparison) == 42
    assert len(pd.read_csv(output / "annual.csv")) == 6*2*12
    assert len(pd.read_csv(output / "monthly.csv")) == 6*2*142
    overall = comparison[comparison.period == "overall"].pivot(
        index="model", columns="strategy", values="mape")
    recent = comparison[(comparison.period == "2025–2026.10") &
                        (comparison.strategy == "periodic")].set_index("model").mape
    earlier = comparison[(comparison.period == "2015–2024") &
                         (comparison.strategy == "periodic")].set_index("model").mape
    selected_scores = pd.DataFrame(selection).set_index("model")
    labels = dict(gam="GAM", lightgbm="LightGBM", nbeats="N-BEATSx",
                  tft="TFT", tcn="TCN", lstm="LSTM")
    lines = ["# 六模型正式实验结果", "",
             "后续探索：[六模型误差分析与 N-BEATSx / LightGBM 等权集成](../error_analysis/README.md)。", "",
             "稳定性复核：[N-BEATSx 三种子跨年回测](../nbeats_seed_repeats/README.md)。", "",
             "本表使用维护者神经架构、限定配置搜索和相同五次定期重训策略。"
             "N-BEATS 家族选中有天气/日历的 N-BEATSx。单位均为 MAPE 百分比。", "",
             "| 模型 | 2014 验证 | 全期冻结 | 全期定期重训 | 2015–2024 定期重训 | 2025–2026.10 定期重训 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in overall.sort_values("periodic").index:
        lines.append(f"| {labels[name]} | {selected_scores.loc[name, 'validation_mape']:.3f}% "
                     f"| {overall.loc[name, 'frozen']:.3f}% | {overall.loc[name, 'periodic']:.3f}% "
                     f"| {earlier[name]:.3f}% | {recent[name]:.3f}% |")
    annual = pd.read_csv(output / "annual.csv")
    baseline_table = pd.DataFrame(baseline_annual).pivot(index="year", columns="lag_days", values="mape")
    baseline_table.columns = ["yesterday", "last_week"]
    order = list(overall.sort_values("periodic").index)
    headings = ["昨天同期", "上周同期"] + [labels[name] for name in order]
    lines += ["", "## 年度 MAPE 对比", "",
              "单位为 %，越低越好；**粗体**为该年八个方案中的最低值，按未四舍五入的数值判定。"
              "两张表列顺序相同，均按全期定期重训成绩排列。2026 年仅含 1—10 月。", "",
              "年度 MAPE 对当年所有半小时的绝对百分比误差求平均，并非年度总用电量误差。"
              "2010–2013 为训练期、2014 为验证期，因此历史回测从 2015 年开始。"]
    for strategy, title, description in [
        ("frozen", "冻结版", "权重与预处理固定在 2010–2014 年拟合结果，每日输入照常更新。"),
        ("periodic", "定期重训版", "2017、2019、2021、2023、2025 年初各重训一次，使用此前四年数据。")]:
        model_table = annual[annual.strategy == strategy].pivot(index="year", columns="model", values="mape")
        table = baseline_table.join(model_table[order])
        table.to_csv(output / f"annual_{strategy}_comparison.csv", index_label="year")
        lines += ["", f"### {title}", "", description, "",
                  "| 年份 | " + " | ".join(headings) + " |",
                  "|---|" + "---:|" * len(headings)]
        winners = []
        for year, row in table.iterrows():
            values = [f"**{value:.3f}**" if value == row.min() else f"{value:.3f}" for value in row]
            year_label = "2026（1—10月）" if year == 2026 else str(year)
            lines.append(f"| {year_label} | " + " | ".join(values) + " |")
            winners.append(headings[list(table.columns).index(row.idxmin())])
        counts = pd.Series(winners).value_counts()
        lines += ["", "年度最低次数：" + "、".join(f"{name} {count} 年" for name, count in counts.items())
                  + "。次数按每年等权计数，不等同于全期 MAPE 排名。"]
    lines += ["", "2015、2016 年尚未发生首次重训，两种策略的成绩相同。"
              "两条 baseline 不涉及训练，因此在两张表中也相同。", "",
              "年度宽表下载：[冻结版](annual_frozen_comparison.csv) · "
              "[定期重训版](annual_periodic_comparison.csv) · [基线年度明细](baselines_annual.csv)。",
              "", "## 评估说明", ""]
    lines += ["", "全期为 2015 年 1 月至 2026 年 10 月，每种策略覆盖 207,456 个半小时。"
              "冻结模型权重保持不变，输入仍每天更新。定期重训在 2017/2019/2021/2023/2025 年初进行，"
              "每次用此前四年数据；超参数固定。", "",
              "神经模型验证成绩为入选配置三个种子的最佳 checkpoint MAPE 均值，"
              "回测固定种子 42，训练轮数取三个最佳轮数的中位数。表格模型为单次固定种子成绩。", "",
              "本轮特征组合因模型而异，搜索预算有限，排名仅适用于本轮预测方案。"
              "近期历史段此前已查看，属于探索性检查，不能宣称为独立未见测试。"
              "最终 live 模型还需验证课程环境兼容性、CPU 推理限制及和队友模型的比较。", "",
              "详细记录：[配置与种子](selection.csv)、[分期比较](comparison.csv)、"
              "[年度](annual.csv)、[月度](monthly.csv)、[官方滞后基线](baselines.csv)、"
              "[结果来源和校验和](manifest.json)。", "",
              "[方法与运行说明](../../../docs/formal_experiment.md)"]
    (output / "README.md").write_text("\n".join(lines) + "\n")
    print(comparison[comparison.period == "overall"].pivot(
        index="model", columns="strategy", values="mape").sort_values("periodic").to_string())
    print(f"Validated six models, two strategies, {len(expected)} half-hours each: {output}")


if __name__ == "__main__":
    main()
