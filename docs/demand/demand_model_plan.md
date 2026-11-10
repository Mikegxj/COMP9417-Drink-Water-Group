# NSW 次日用电需求预测实验设计

> 2026-11-08 更新：下文保留最初设计与紧凑实现的历史结果。当前维护者架构、七模型配置搜索及统一定期重训协议见[正式实验说明](formal_experiment.md)；完成并核验后的统一结果写入 results/demand/formal_report。请勿将下文初筛或旧适应成绩与正式实验混合排名。
>
> 2026-11-10：M5′ 已移除（m5py 与课程环境 scikit-learn 1.9.1 不兼容）。下文对 M5′ 的描述仅为历史记录，表中 M5′ 行已删除，代码不再支持 `m5`。见 [正式实验说明](formal_experiment.md)。

目标是在每天结束时预测下一市场日的 48 个半小时 NSW 需求（MW），按历史回测 MAPE 选出一个可部署的 live 需求模型。个人负责七个候选，队友负责 Ridge 和 MLP，全组共用预测日期、两个官方基线和评分代码。电价方向由独立模块输出。

## 已有结果与下一场比较

用 2010–2013 年训练，2014 年全年验证，共 17,520 个半小时。以下为已完成的初试，其余模型待运行。两种模型均使用前 1、2、7 天同期负荷、悉尼目标时段及前一天同期气温、星期和年内周期。GAM 按时段分别拟合，M5′ 使用显式时段列。[初试脚本](../../experiments/demand/pilot/pilot_gam.py)

| 2014 验证 | MAPE（越低越好） | MAE（MW） |
|---|---:|---:|
| GAM，按时段拟合的样条加性模型 | 2.582% | 204.3 |
| 昨天同时段 | 4.935% | 394.4 |
| 上周同时段 | 4.934% | 401.3 |

下一步先让 LightGBM 使用初试同一份输入，与 GAM、M5′ 直接比较。保留这三行模型配置后再引入假期和其他天气变量，确保每次改善都能追溯到具体改动。

## 每天预测时能看到什么

市场时间为 AEST（UTC+10）。区间结束时间从 00:30 到次日 00:00 构成一个市场日。以 D 日结束为预测起点，一次生成 D+1 日全部 48 点。训练样本也按相同的预测起点构造。

| 信息 | 取值规则 |
|---|---|
| 历史负荷 | 表格模型按目标时段取同期滞后。序列模型取截至 D 日结束的连续负荷。计算历史统计量时，最后一个可用市场日为 D。所有州的 D+1 负荷和价格均从输入中排除。 |
| 天气 | 官方接口提供 D+1 的 48 行天气。Sydney、Melbourne、Brisbane 各有气温、相对湿度、云量、短波太阳辐射。课程将再分析天气视为已知，评估结果应注明这一假设。 |
| 日历 | 从目标市场日计算时段、星期、年内周期，可加入课程允许的 NSW 公共假期。午夜结束的区间沿用前一市场日的日历。 |
| 预处理 | 仅在训练窗口拟合编码、缩放、样条节点与特征筛选。训练、验证和测试按市场日切分。所有模型在完全相同的验证日期评分，早停仅使用当时可用的验证数据。 |

数据仅使用课程 CSV 和允许的日历信息。[数据字典](../../data/README.md) · [市场日与指标实现](../../harness/dayahead.py)

## 七个候选模型如何使用输入

| 模型 | 第一版设计 | 优先验证的问题 |
|---|---|---|
| GAM | 沿用初试的 48 个时段模型。温度用样条，负荷滞后用线性项，系数带 Ridge 惩罚。 | 加入前一个完整市场日的最高、最低温，能否改善连续高温天气的预测？ |
| LightGBM | 每个目标半小时一行，先直接预测 MW。 | 先将滞后由前 1、2、7 天扩至前 1–7 天各自同期。可再单独试预测相对昨天同期的残差，加回基线后计算 MAPE。 |
| N-BEATS | 基础版读过去 336 点负荷，直接输出次日 48 点。 | 若测试天气，使用支持未来外部变量的 N-BEATSx。对同一 N-BEATSx 配置开关气温，区分架构变化与天气贡献。 |
| TFT | 分开传入历史 336 点负荷与次日已知的 48 点气温和日历，预测全天。 | 固定历史窗口，比较加入悉尼气温前后误差。 |
| TCN | 因果卷积编码历史负荷，结合每个目标时段的已知输入输出 48 点。 | 先用 336 点，再试 912 点。同步检查卷积感受野确实覆盖所比较的窗口。 |
| LSTM | 编码过去 336 点负荷，将历史表示与各目标时段气温、日历组合，一次输出全天。 | 初筛用 7 天；入围后可试 10 天（480 点），固定其他配置。 |

队友的 Ridge 与 MLP 接入同一评估流程。每个模型登记实际使用的特征和窗口，再比较其最佳可部署版本。完整序列与少数同期滞后包含的信息量不同，跨模型排名用于选预测方案；解释某项特征的贡献时，只比较同一模型的配对实验。

## 特征依据与实验顺序

| 设计依据 | 本项目的使用方式 |
|---|---|
| 澳洲半小时负荷研究以日历、历史需求、历史及预测气温建立加性模型。 | GAM 分时段建模，并检验历史温度摘要。[Fan 与 Hyndman](https://robjhyndman.com/publications/stlf/) |
| NSW 模型树研究使用历史负荷、时段及 Sydney 气温、湿度、风速；另有日前研究在 LightGBM 阶段使用过去七天的同期负荷。 | M5′ 使用官方已有的温湿度。LightGBM 单独比较三项与七项同期滞后。[NSW 模型树研究](https://www.mdpi.com/1996-1073/16/2/867) · [LightGBM 研究](https://pmc.ncbi.nlm.nih.gov/articles/PMC8623363/) |
| 澳洲 N-BEATS 实验以 336 点预测 48 点。葡萄牙五模型研究分别调历史窗口，并对照纯负荷与加入气温的结果。 | 统一从一周历史起步，将论文窗口当作候选。架构和天气配置以 NSW 验证结果决定。[澳洲 N-BEATS 研究](https://pubs2.ascee.org/index.php/IJRCS/article/view/2437) · [葡萄牙日前负荷研究](https://arxiv.org/html/2302.12168v2) |
| 负荷预测已有采用云量、辐射及选择多个气象站的研究。 | 分别测试 Sydney 的云量和辐射。墨尔本、布里斯班对 NSW 的额外价值单独验证。[住宅负荷研究](https://www.mdpi.com/2624-6511/8/6/199) · [气象站选择研究](https://www.sciencedirect.com/science/article/pii/S0169207014001319) |

先完成所有模型的一版。仅对 2014 年入围的两至三个模型深入调参，依次回答以下问题：

1. **次日气温贡献多少？** 固定历史负荷、历史天气（如有）、日历和训练设置，仅开关次日悉尼气温。纯负荷 N-BEATS 另列。
2. **哪些天气值得保留？** 在气温版上分别加湿度、云量、太阳辐射，再试云量与辐射同时加入。比较全年 MAPE 以及夏季午后误差。
3. **其他城市还能补充什么？** 固定已选的 Sydney 特征，分别加入 Melbourne、Brisbane 的同类天气。保留有验证收益的组合。每个对照记录字段、窗口、种子、训练耗时、逐点预测及分组误差。

## 回测、适应与 live 选型

1. **2014 年初选。** 用 2010–2013 训练。所有候选与昨天同期、上周同期两条基线在相同日期比较 MAPE 和 MAE。按 2014 年 MAPE 选择可部署的需求模型进入适应实验，接近时优先选 CPU 推理更快、重建更简单者。固定各模型配置后，可用 2010–2014 重训得到冻结版本。
2. **2015 年 1 月至 2026 年 10 月逐日模拟。** 对第一阶段研究过的每种模型评估其冻结版本。对选定模型比较冻结、每两年重训、漂移触发重训三种策略；后两种先统一采用最近四年窗口。每个模型与策略组合最多重训 6 次，更新拟合型预处理计入同一次重训。输入漂移监测可用最近 28 日的日负荷曲线与固定训练期聚类中心的平均距离，误差检测可用最近 28 日平均日 MAPE。阈值在 2010–2014 校准，记录信号日与重训日，D 日结束触发的更新最早用于 D+1。
3. **报告与选择。** 按年、月分别报告冻结与适应版本，对照各期两条基线。画出典型周预测曲线、月度 MAPE 及两种 95% 区间（半小时独立近似、按完整市场日 bootstrap 至少 1,000 次）。仅在 2010–2013 拟合日曲线 PCA 与聚类，固定后分析各年日型占比与误差变化。预留 2025 年至 2026 年 10 月作近期检查，打开前锁定候选；建议以此前历史模拟整体 MAPE 排序，并单独报告近期表现。最终部署在历史模拟中表现最佳且满足提交约束的方案，记录选择依据。
4. **最终训练。** 按入选策略的训练窗口，使用截至 2026 年 10 月 31 日的允许数据重建提交模型。live week 每天读取新历史计算特征，拟合参数保持固定。

## 计算与提交

本机先跑表格模型、序列数据检查和各神经网络的短训练。以实测每轮耗时估算剩余时间，再决定是否将长训练发往 Katana；排队期间本机继续其他模型。比较环境间的配置、种子和数据切分。

M5′ 初试依赖 m5py 0.3.3，课程环境未列入该包。若其入围，须先验证在指定环境中可重建和推理；使用额外神经网络库的候选也需完成同一检查。最终训练脚本只重建提交模型，全组包离线重建限时 2 小时。与电价模块合并后，在指定 Linux CPU 环境验证加载 ≤60 秒、每日预测 ≤20 秒、内存 ≤4 GB、模型文件合计 ≤100 MB。需求输出列为 nsw_demand_mw，恰好 48 个有限值，范围 0–20,000 MW，索引与 weather_next 完全一致。[作业说明](../../project_spec.pdf) · [提交接口](../../template/model.py)

## 首轮实现与复现

七个候选的 2014 初筛与 2015–2026 年 10 月冻结回测已完成。以 2014 年最佳变体进入历史模拟，M5′ 平分时保留基础输入。四个神经模型采用单种子、16 轮的紧凑 PyTorch 点预测实现；TFT 为单序列、单头注意力版本。N-BEATSx 的 calendar_temp 变体在通用残差块上加入已知日历和气温的预测基函数。模型名 nbeats 的输出文件须结合 variant 区分基础版和扩展版。

| 模型与 2014 入选配置 | 2014 MAPE | 冻结 2015–2024 MAPE | 冻结 2025–2026.10 MAPE |
|---|---:|---:|---:|
| LightGBM，加 Sydney 云量 | 2.483% | 3.969% | 7.058% |
| N-BEATSx，日历与 Sydney 气温 | 2.507% | 6.199% | 15.009% |
| GAM，加前日最低及最高温 | 2.519% | 3.892% | 6.050% |
| LSTM | 2.765% | 4.481% | 7.299% |
| TFT | 2.779% | 6.488% | 13.453% |
| TCN | 2.847% | 4.851% | 7.580% |

按 2015–2024 选策略，LightGBM 每两年使用最近四年重训的 MAPE 为 3.396%，在近期检查段为 4.550%。误差触发策略对应 3.600%、6.526%，到 2020 年就已用完六次更新。此处仅完成 LightGBM 内部策略比较。跨模型的同策略结果见下节。近期检查段已打开，未来根据这些结果做的修改需要标注为后续探索。现有模型文件仅供本地需求模块联调，提交前在课程 Python 3.11/Linux 环境重建并合入电价模块。

从仓库根目录运行以下命令。本机使用项目虚拟环境中的 Python 3.13、PyTorch 2.10、LightGBM 4.7.0、scikit-learn 1.7.2；M5′ 初试另需 m5py 0.3.3。下面的 python 指向已安装依赖的解释器。

```sh
python -m experiments.demand.screen.run_demand --stage screen
python -m experiments.demand.screen.run_demand --models nbeats --variant calendar_only
python -m experiments.demand.screen.run_demand --models nbeats --variant calendar_temp
for v in no_target_temp prior_extrema seven_lags humidity cloud solar cloud_solar mel_temp bne_temp; do
    python -m experiments.demand.screen.run_demand --models gam lightgbm --variant "$v"
done
python -m experiments.demand.screen.run_demand --stage frozen --models lightgbm --variant cloud
python -m experiments.demand.screen.run_demand --stage frozen --models gam --variant prior_extrema
python -m experiments.demand.screen.run_demand --stage frozen --models nbeats --variant calendar_temp
python -m experiments.demand.screen.run_demand --stage frozen --models lstm tft tcn
python -m experiments.demand.early_adaptation.adapt_demand
python -m experiments.demand.screen.summarize_demand
python -m experiments.demand.early_adaptation.live_demand
```

[初筛各变体](../../results/demand/screen/comparison.csv) · [策略及检查期比较](../../results/demand/historical_comparison.csv) · [年度指标](../../results/demand/annual.csv) · [月度指标及置信区间](../../results/demand/monthly.csv)

## 七模型同策略复核

沿用 2014 年入选配置，所有模型初始训练截至 2014 年底，分别在 2017、2019、2021、2023、2025 年初用此前四年的样本重训。神经网络固定 16 轮和种子 42。七个模型均在相同的 207,456 个半小时上预测。按 2015–2024 年 MAPE 排序，另列已查看过的近期时段作为描述性检查。

| 模型 | 2015–2024 MAPE | 2025–2026.10 MAPE | 全期 MAPE |
|---|---:|---:|---:|
| LightGBM | 3.396% | 4.550% | 3.574% |
| N-BEATSx | 3.555% | 5.237% | 3.815% |
| GAM | 3.580% | 5.150% | 3.823% |
| LSTM | 3.901% | 5.529% | 4.153% |
| TCN | 4.098% | 5.604% | 4.331% |
| TFT | 4.719% | 5.284% | 4.806% |

当前 LightGBM 是个人七个已测配置中领先者。小组 live 选型待 Ridge 和 MLP 在同一协议下比较；漂移触发策略目前只对 LightGBM 评估。复现顺序为先运行 `python -m experiments.demand.early_adaptation.periodic_demand`，再运行 `python -m experiments.demand.screen.summarize_demand`。[定期重训明细](../../results/demand/periodic/comparison.csv)
