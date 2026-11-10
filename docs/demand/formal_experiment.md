# 六模型正式配置搜索与历史回测

> 2026-11-10：**M5′ 已从候选模型中移除。** 它依赖的 m5py 与课程固定的 scikit-learn 1.9.1 不兼容，无法在评分环境中运行；在任何阶段它都不是领先模型（Stage 1 exp2 MAPE 2.924%，差于 GAM 2.283%、LightGBM 2.294% 和两个神经模型）。相关代码、结果文件、结果行与表格列均已删除。
>
> 同日：课程组拒绝在评分环境加入 neuralforecast（依赖过重，与沙箱限制冲突），允许用纯 PyTorch 重写架构。N-BEATSx 与 TFT 现由 `src/demand/torch_models.py` 提供，详见文末“纯 PyTorch 重写”。

执行日期为 2026 年 11 月 8 日。该轮落实架构核对、配置搜索、随机种子复核以及重新运行跨年回测。结果写入 results/demand/formal。

**完成状态：六模型配置搜索、冻结回测及相同五次定期重训回测均完成。** 统一结果见[六模型结果](../../results/demand/formal_report/README.md)。每个模型每种策略覆盖 207,456 个半小时，年度共 144 行、月度共 1,704 行，已核验完整日期、真实值、训练边界和结果来源。所有预测在 0–20,000 MW 范围内。神经模型入选配置的各个种子均达到耐心早停条件。

本轮全期定期重训 MAPE 最低的是 N-BEATSx（3.445%），其次为 LightGBM（3.528%）；近期段 LightGBM 为 4.490%，低于 N-BEATSx 的 4.689%。冻结版全期最低为 GAM（4.226%）。排序会随策略和时间段变化，不能把 LightGBM 或任何模型称为无条件第一。最终提交选择还需与队友的 Ridge/MLP 比较，并检查课程环境与运行限制。

## 后续探索：误差分析与等权集成

已完成六模型两种策略的月份、半小时时段、星期类型及气温区间误差分析，并以固定 50/50 权重组合 N-BEATSx 与 LightGBM 的逐点预测，未重新训练或搜索权重。[完整诊断与结果](../../results/demand/error_analysis/README.md)。运行 `python -m experiments.demand.error_analysis.analyze_demand_errors` 可重现；程序检查预测日期、真实值、集成绝对误差的凸性上界，以及各分组按样本数加权后恢复总体 MAPE。

定期重训集成全期 MAPE 为 3.076%，近期为 4.063%；142 个月中 121 个月同时优于两个组成模型，近期为 20/22 个月。冻结集成全期为 5.084%，未超过冻结 LightGBM。集成仅作新增探索方案，不替换原六模型表。输入模型仍来自本轮种子 42 的回测；多种子回测、训练窗口对照、新特征与扩大搜索尚未在这项分析中执行。

## N-BEATSx 三种子回测复核

已补完种子 43、44 的冻结版和五次定期重训版，种子 42 复用原结果。所有运行固定 480 点历史、宽度 128、天气/日历、学习率 0.001、59 轮和原训练区间；环境保持 Python 3.13.9、Torch 2.10.0、NeuralForecast 3.3.0 的本机 CPU。运行 `python -m experiments.demand.backtest.repeat_nbeats_backtest` 可复现或从已完成拟合继续。[完整种子、年度和月度结果](../../results/demand/nbeats_seed_repeats/README.md)。

定期重训全期 MAPE 分别为 3.445%、3.471%、3.535%，均值 3.484%，样本标准差 0.046 个百分点。近期分别为 4.689%、4.820%、4.838%，均值 4.782%，标准差 0.082 个百分点。对照现有 LightGBM（全期 3.528%、近期 4.490%），N-BEATSx 全期有两个种子更低、一个更高，近期三个种子均更高。不能据单次种子 42 宣称全期稳定领先；三种子均值也不是集成预测成绩。

已验证同种子可复现、不同种子改变拟合、错误种子检查点被拒绝；完成结果核验包括数据与环境一致性、逐点真实值/日期、有限值、检查点配置与种子、训练区间和首次更新前策略一致性。输出共 18 条分期指标、72 条年度、852 条月度记录。后续窗口或特征变动应使用同样种子配对比较。本轮未重新选配置、种子或集成权重。

## 模型及训练方法

GAM 使用分时段样条加性回归，LightGBM 使用官方 LGBMRegressor。N-BEATSx 与 TFT 使用 `src/demand/torch_models.py` 中的纯 PyTorch 实现（见下文“纯 PyTorch 重写”），TCN 与 LSTM 仍直接实例化 NeuralForecast 3.3.0。均只使用模型的 forward 方法，外层由项目代码控制 Adam、训练窗口和早停；库内的自动训练、窗口缩放和调度流程未启用。目标损失为 MAE，选择指标为还原到 MW 后的 MAPE。

N-BEATS 家族包含纯负荷 NBEATS 与已知天气/日历输入的 NBEATSx。TFT 保留维护者实现的变量选择、GRN、LSTM 和可解释注意力。数据只有单一 NSW 序列，因此不额外构造静态协变量。TCN 的 kernel=3，膨胀层数根据窗口长度选择，卷积感受野覆盖窗口。LSTM 使用维护者的多步解码架构。相较旧实现，各模型的结构及输入通路发生变化，结果应作为新实验记录。

每个市场日一个训练样本，历史窗口为 336 或 480 个半小时，输出未来 48 点。除纯负荷 NBEATS 外，神经模型接收历史及次日的 Sydney 气温、时段/星期/年内日期周期编码。缩放均值和标准差只在训练区间计算。没有未来需求或价格进入 forward 输入。

天气遵循课程接口：将提供的次日天气视为已知。CSV 中天气并非按历史发布时刻存档的业务天气预报，因此这些成绩不衡量真实天气预报误差传递到负荷预测的影响。

表格模型使用前 1/2/7 天同时段负荷、前一天同时段气温、次日气温与日历。GAM 另有前一天最高/最低气温，LightGBM 另有次日云量。这些特征版本沿用此前 2014 年验证选定的版本，本轮固定特征后搜索模型配置。因此本轮比较的是各自特征管道加模型的效果；不能把差异完全归因于网络结构，也不是六模型统一天气特征的消融实验。GAM 是项目构建的分时段样条加性回归，并非 Fan–Hyndman 论文完整复现。

架构依据为 [NeuralForecast NBEATSx](https://nixtlaverse.nixtla.io/neuralforecast/models.nbeatsx.html) 和 [NeuralForecast TFT](https://nixtlaverse.nixtla.io/neuralforecast/models.tft.html)。完整候选与环境版本保存于 [执行协议](../../results/demand/formal/protocol.json)。

## 配置预算与选择

- GAM 12 组，样条节点 3/5/7/9，alpha 为 0.1/1/10。
- LightGBM 12 组，叶数 15/31/63，叶节点最小样本数 24/48，学习率 0.02/0.04，500 棵树。
- 每个神经家族 8 组。N-BEATS 家族交叉比较窗口、外部变量开关、学习率；其他神经模型交叉比较窗口、16/32 隐藏宽度、学习率 0.001/0.002。

神经配置先分配 32 轮，按最佳验证 MAPE 保留两组。每组用 42/43/44 三个种子训练，上限 160 轮；若尚未满足早停，延长到 320 轮。连续 20 轮验证 MAPE 未改善则停止，恢复最佳权重。对两组按三个种子的最佳验证 MAPE 均值选配置，保存标准差。32 轮初筛可能筛掉收敛较慢的配置，因此结论限于本轮预算和范围。任何达到最终上限的运行单独标记。

只使用 2010–2013 年训练和 2014 年验证。神经模型重训轮数使用入选配置三个种子最佳轮数的中位数。后续历史误差不用于修改本轮配置。

## 历史回测

每个入选配置使用 2010–2014 年重训，得到冻结模型，逐日预测 2015 年 1 月至 2026 年 10 月。同期对每个模型执行相同定期重训，在 2017、2019、2021、2023、2025 年初使用此前四年的训练样本更新。神经模型固定种子 42 和在 2014 年选出的重训轮数。每次重新拟合缩放，与模型一同更新。

冻结指模型权重和缩放参数不更新。两种策略每天都会使用截至前一日已观测到的真实历史负荷，以及下一日允许使用的天气；不把多年前的预测递归当作整个十余年的输入。定期重训更新权重/缩放，超参数与神经训练轮数维持 Stage 1 选择值。

保存逐半小时预测、年度及月度 MAPE/MAE、月度半小时独立近似区间和整日 bootstrap 区间。分别汇总 2015–2024、2025–2026.10 和全期。后两年已在前轮查看过，因此本轮近期成绩作为探索性历史检查报告。

## 运行与检查

表格模型与神经库分开进程运行，避免本机 OpenMP 运行库混合加载引发异常。每次神经训练保存 last.pt、优化器状态和随机数状态，可从已完成轮数继续。神经模型的重训权重及缩放参数另保存在回测目录。文件路径在 results/demand/formal 下，记录与以前的简化架构结果分别保存。

```sh
python -m experiments.demand.config_search.tune_demand --models gam lightgbm
python -m experiments.demand.config_search.tune_demand --models nbeats tft tcn lstm
python -m experiments.demand.backtest.formal_backtest --models gam lightgbm
python -m experiments.demand.backtest.formal_backtest --models nbeats tft tcn lstm
python -m unittest experiments.demand.config_search.test_formal_models
```

边界验证包括 336/480 点窗口与正式逐日接口对齐、改变验证期真实需求后首日输入/输出不变、维护者模型输出形状与梯度有限。正式模型三项与原有边界六项检查均已通过。新增依赖属于实验环境，课程评分环境的依赖与模型导出仍需在最终提交阶段验证。

## Katana 执行

用户明确授权上传 experiments/、harness/ 和课程 CSV 到其 Katana scratch（作业脚本含个人路径，未入库）。首次作业 9701137 因环境依赖与 V100 的 CUDA 构建不匹配而失败，未产生有效模型结果；修复后作业 9701171 在节点 k101 的 V100 上正常训练，负责 TFT、TCN 的完整搜索及回测，输出到远端 results/demand/formal_gpu。若该作业完成且检查通过，本轮 TFT、TCN 以该 GPU 执行为准，原因是计算耗时；此决定在取得 GPU 完整结果前作出。本机已有 CPU 中间结果保留供审计，重复 TFT 训练已停止。GPU 使用 Python 3.11、NeuralForecast 3.3.0、Torch 2.14.1+cu126，本机版本见 protocol.json。跨硬件随机数和浮点运算差异纳入解释。

完整结果回传后运行 `python -m experiments.demand.backtest.report_formal`，验证六个模型各自两种策略的时间戳、真实值、有限预测、五次重训边界，以及首次重训前两种策略预测一致。该命令在 results/demand/formal_report 生成统一 comparison.csv、annual.csv、monthly.csv、selection.csv 与来源及校验和 manifest.json，不覆盖原始 CPU/GPU 实验。月度整日 bootstrap 将各日作为独立抽样单位，未消除跨日自相关，其区间仅作描述性参考。

作业 9701171 最终正常退出（Exit_status=0），运行 32 分 36 秒。上述结果已回传并执行汇总核验。完整 GPU 训练日志、依赖版本与执行信息位于 results/demand/formal_gpu；本轮 TFT/TCN 采用该目录的完成结果。本机中断的 TFT/TCN 初筛文件不参与排名。

### LightGBM 种子补充核验

对当前固定 LightGBM 配置，在全部六个拟合窗口补跑种子 43、44，与种子 42 的逐点预测最大差异约 1.82e-12 MW，属于 CSV 浮点读写精度量级。当前 subsample=1、subsample_freq=0、colsample_bytree=1，没有启用行或列随机采样；实测更换这些种子没有可测影响。此结论仅限当前配置、数据与环境，不能推广到开启随机采样后的 LightGBM。脚本为 experiments/demand/backtest/check_lightgbm_seeds.py，记录见 [逐窗口核验](../../results/demand/lightgbm_seed_check/comparison.csv)。

## Stage 1 补充：公众假日特征消融（2026-11-10）

在 2014 验证上（2010–2013 训练）检验 NSW 公众假日特征，来源为课程环境的 holidays 包（排除仅银行放假的 Bank Holiday）。各模型固定原 Stage 1 入选配置，只改特征。逐级叠加：exp1 加当天是否假日 `hol_today`；exp2 再加 `hol_lag1`、`hol_lag7`（滞后值来自假日的标记）；exp3 再加 12 月 24 日至 1 月 2 日的非法定假日标记。选择规则事先固定：2014 全年 MAPE 最低；与更简单版本的配对整日 bootstrap 区间含 0 时选简单版本。

结果：GAM、LightGBM 的 exp3 相对 exp2 区间均含 0，六个可比模型统一采用 **exp2**。表格模型用课程固定版本环境，神经模型在同一 CPU 环境下以种子 42/43/44 同时重跑基线与 exp2 作配对比较（该环境基线与原 Mac/GPU 结果存在跨硬件差异，因此不与原数字直接相减）。TCN、LSTM 未加入（排名靠后、代价高）；纯负荷 NBEATS 不使用外生变量。

| 模型 | exp0 | exp2 | 假日当天 exp0→exp2 |
|---|---|---|---|
| N-BEATSx（3 种子均值±标准差） | 2.119±0.040 | **1.887±0.040** | 8.47→2.67 |
| TFT（3 种子均值±标准差） | 2.130±0.052 | **1.914±0.047** | 7.69→2.90 |
| GAM | 2.519 | 2.283 | 10.14→4.87 |
| LightGBM | 2.446 | 2.294 | 7.44→2.99 |

2014 仅 11 个假日、14 个假日后受影响日、5 个圣诞期日，分组结果只作诊断。现有 2015–2026 回测均为旧特征，需用 exp2 重跑。复现：`python -m experiments.demand.holiday_ablation.holiday_ablation`，`python -m experiments.demand.holiday_ablation.holiday_ablation_neural`。结果位于 results/demand/holiday_ablation、holiday_ablation_neural。

## Stage 2：exp2 特征的历史回测与漂移触发重训（2026-11-10）

用 exp2 特征（假日三项标记）重跑 2015-01 至 2026-10 逐日模拟，结果写入 results/demand/formal_exp2，不覆盖旧结果。各模型固定 Stage 1 配置；神经模型重训轮数取 exp2 三种子最佳轮数中位数（N-BEATSx 36、TFT 43），种子 42。GAM、LightGBM、N-BEATSx 跑冻结与五次定期重训；TFT 只跑规范要求的冻结版。TCN、LSTM 的 Stage 1 特征未变，沿用 formal_report 的旧冻结结果。

| 模型·策略 | 2015–2024 | 2025–2026.10 | 全期 |
|---|---|---|---|
| N-BEATSx 冻结 | 4.775 | 11.070 | 5.749 |
| N-BEATSx 定期重训（5 次） | **3.060** | 4.523 | **3.287** |
| LightGBM 冻结 / 定期 | 3.746 / 3.222 | 6.752 / **4.441** | 4.211 / 3.410 |
| GAM 冻结 / 定期 | 3.711 / 3.444 | 5.895 / 5.080 | 4.049 / 3.697 |
| TFT 冻结 | 5.891 | 13.677 | 7.096 |

对照基线（昨天同期）：2015–2024 5.897、近期 7.533、全期 6.150。冻结神经模型近期段劣于该基线。

**漂移触发重训（N-BEATSx，`python -m experiments.demand.drift_retrain.adapt_exp2`）。** 检测器为最近 28 日平均日 MAPE；阈值取 2010–2013 训练、2014 预测的同配置模型在 2014 上滚动值的分位数，主设定 q=0.95 事先固定，0.90/0.99 为敏感性分析。D 日信号以截至 D 日的最近四年数据重训，D+1 起生效；最多 6 次，两次重训至少相隔 180 天；初始拟合与冻结版共用同一检查点。

| 策略 | 重训次数 | 最后一次重训 | 2015–2024 | 2025–2026.10 | 全期 |
|---|---|---|---|---|---|
| 冻结 | 0 | — | 4.775 | 11.070 | 5.749 |
| 定期（每两年） | 5 | 2025-01 | **3.060** | **4.523** | **3.287** |
| 误差漂移 q=0.90（阈值 2.37%） | 6 | 2018-09 | 4.480 | 10.422 | 5.400 |
| 误差漂移 q=0.95（阈值 2.80%） | 6 | 2019-09 | 3.813 | 8.397 | 4.523 |
| 误差漂移 q=0.99（阈值 2.98%） | 6 | 2020-04 | 3.366 | 6.656 | 3.875 |

三个阈值都在 2020 年 4 月前用完 6 次预算，之后模型冻结老化，全期均劣于定期重训。原因是阈值来自刚训练完的模型在紧接一年上的误差，低于部署中模型的正常误差水平，规则实际退化为"每满 180 天重训一次"。2018–2020 年漂移策略反而优于定期（如 q=0.99 在 2018、2020 年为 2.34%、3.14%，定期为 2.61%、3.71%），说明更早重训有用，问题在阈值校准与预算分配。逐年结果见 adaptation/nbeats/strategy_annual.csv。

## 纯 PyTorch 重写（2026-11-10）

课程组回复：neuralforecast 依赖过重，与评分沙箱的限制冲突，不加入环境；允许用纯 PyTorch 重写架构。`src/demand/torch_models.py` 按 neuralforecast 3.3.0 源码重写了实际使用的部分：N-BEATSx（3 个 identity 栈，未来外生变量）与 TFT（未来外生变量、LSTM 编码器，无静态/历史外生变量）。层顺序、初始化顺序和参数名与原库一致，原有 state_dict 可直接加载。`src/demand/neural.py` 中 `make_model` 的 `nbeats`/`tft` 已改用该模块；TCN、LSTM 只用于早期筛选，仍延迟导入 neuralforecast。

核验：

- `python -m experiments.demand.torch_rewrite.check_torch_models`（需另装 neuralforecast，仅开发用）：5 组配置下初始权重、前向输出、含 dropout 的 5 步训练后参数，以及跨实现加载后的预测，差异均为 0（逐位一致）。
- 在课程环境（Python 3.11、requirements.txt 固定版本 + torch 2.15.0，未装 neuralforecast）中，把原库训练的权重载入新模块：N-BEATSx 冻结回测（2010–2014 训练，预测 2015-01 至 2026-10，207,456 个半小时）MAPE 6.7075%，与存档一致，逐点最大差 0.008 MW；TFT Stage 1 检查点 config_3 / config_7 的 2014 MAPE 2.2798% / 2.2626%，与存档一致，最大差 0.002 MW。差异来自 float32 与 CSV 舍入。
- `python -m unittest experiments.demand.config_search.test_formal_models` 在课程环境中 3/3 通过。

因此已有 exp2 结果无需重跑，可直接沿用；提交的 model.py 只需包含 torch_models.py 中对应类并加载 state_dict。
