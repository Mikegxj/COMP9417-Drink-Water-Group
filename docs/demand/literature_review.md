# 文献综述：NSW 次日负荷预测相关研究

> 2026-11-08 记录。用于支撑[需求模型设计](demand_model_plan.md)和[正式实验](formal_experiment.md)的模型与特征选择。来源已于同日检索核对，完整引用见文末[参考文献](#参考文献)。

## 研究汇总

| 研究 | 模型与发现 | 对本项目的意义 | 来源 |
|---|---|---|---|
| 澳洲国家电力市场，半小时负荷预测（Fan & Hyndman） | 半参数加性模型，结合历史需求、气温和日历，预测未来最多七天的半小时需求。 | 地区背景、采样频率和输入都很接近。可借鉴它研究气温的非线性影响：20→25°C 与 30→35°C 对负荷的影响差别有多大。 | [1] |
| NSW，日前负荷预测 | 使用 NSW 半小时负荷和 Sydney 天气，以 M5P 模型树预测，并用随机森林和遗传算法筛选特征。模型树按条件划分样本，再在叶节点做回归。 | 数据背景最贴近本项目，也为尝试树模型提供了直接依据。可用 LightGBM 检验昨天和上周同时段负荷、次日气温各自带来多少改善。 | [2] |
| 葡萄牙，全国日前负荷预测 | 同一实验比较 MLP、LSTM、N-BEATS、TCN 和 TFT。只用历史负荷时，N-BEATS 测试 MAPE 1.90%，MLP 2.02%；加入里斯本温度的事后实验中，MLP 1.92%，N-BEATS 2.00%（均为集成结果）。 | 对模型选择最有帮助：同一批数据仅改变可用输入，排名就发生变化。队友的 MLP 值得认真调；序列模型也应单独测量加入天气后的收益。 | [3]，表 4 |
| NSW 和 Queensland，直接预测次日 48 点 | N-BEATS，输入过去 7 天 × 48 = 336 个点，输出未来 48 个点。用最近 28 天训练、最后一天测试，并比较随机搜索、TPE 和遗传算法调参。 | 可直接借鉴输入输出设计。本项目已有跨年历史回测，能进一步回答它在夏季高温、节假日和不同年份是否稳定。 | [4] |
| 太阳能，日前光伏出力预测 | 一项研究在匈牙利 16 个光伏电站、两年期 15 分钟数据上比较 24 种模型，输入包含数值天气预报，最好的是核岭回归和 MLP，加入太阳位置角和辐照度衍生量使 RMSE 下降 13.1%。另一项在纽约州 100 个电站的比较中，LightGBM 的 RMSE 最低（6.07），线性回归也有竞争力（7.27）且最快。 | 最值得借鉴的是如何比较天气输入：固定模型，依次加入次日温度、云量和太阳辐射，看哪些变量真正改善 NSW 下午负荷预测。 | [5]；[6] |

## 对实验的启示

1. **气温非线性**：参考 Fan & Hyndman，检查高温区间的边际影响，GAM 样条已部分覆盖，可加分段或交互项。
2. **特征消融**：固定 LightGBM / 模型树，逐项加入昨天同时段、上周同时段、次日气温，记录 MAPE 变化。
3. **模型排名依赖输入**：MLP 与序列模型（N-BEATS 等）都要在"仅负荷"和"负荷 + 天气"两种输入下分别报告。
4. **稳定性**：沿用跨年回测，按夏季高温日、节假日和年份拆分误差，而不是只看单一测试期。
5. **天气变量筛选**：如数据可得，依次测试温度、云量、太阳辐射对下午时段（屋顶光伏影响最大）的作用。

## 参考文献

1. S. Fan, R. J. Hyndman. Short-Term Load Forecasting Based on a Semi-Parametric Additive Model. *IEEE Transactions on Power Systems*, 27(1):134–141, 2012. doi:[10.1109/TPWRS.2011.2162082](https://doi.org/10.1109/TPWRS.2011.2162082)。工作论文版：[Monash EBS WP 17/10](https://econpapers.repec.org/RePEc:msh:ebswps:2010-17)
2. A. K. Srivastava 等. A Day-Ahead Short-Term Load Forecasting Using M5P Machine Learning Algorithm along with Elitist Genetic Algorithm (EGA) and Random Forest-Based Hybrid Feature Selection. *Energies*, 16(2):867, 2023. [MDPI 原文](https://www.mdpi.com/1996-1073/16/2/867)，doi:[10.3390/en16020867](https://doi.org/10.3390/en16020867)
3. S. Pelekis 等. A comparative assessment of deep learning models for day-ahead load forecasting: Investigating key accuracy drivers. [arXiv:2302.12168](https://arxiv.org/abs/2302.12168), 2023.
4. Optimizing N-BEATS for Short-Term Load Forecasting Via Random Search–TPE and Genetic Algorithm. *International Journal of Robotics and Control Systems*. [文章页](https://pubs2.ascee.org/index.php/IJRCS/article/view/2437/0)
5. D. Markovics, M. J. Mayer. Comparison of machine learning methods for photovoltaic power forecasting based on numerical weather prediction. *Renewable and Sustainable Energy Reviews*, 161:112364, 2022. doi:[10.1016/j.rser.2022.112364](https://doi.org/10.1016/j.rser.2022.112364)
6. Toward better integration of solar energy in transportation systems: machine learning benchmarks for day-ahead photovoltaic power forecasting. [TRID 记录](https://trid.trb.org/View/2705498)
