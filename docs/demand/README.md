# 需求预测（Cattie 负责部分）

| 位置 | 内容 |
|---|---|
| [formal_experiment.md](formal_experiment.md) | **主文档**：Stage 1 配置搜索、假日特征消融、Stage 2 回测与漂移触发重训、纯 PyTorch 重写 |
| [experiment_log.md](experiment_log.md) | **实验日志**：按时间顺序列出每次实验的设置、结果与决定 |
| [demand_model_plan.md](demand_model_plan.md) | 最初设计与紧凑实现的历史结果（历史记录） |
| [model_audit.md](model_audit.md) | 早期实现的收敛审查（历史记录） |
| [literature_review.md](literature_review.md) | 文献综述 |
| `src/demand/` | 可复用的模型代码：`tabular.py`（GAM、LightGBM 特征与模型）、`neural.py`（日窗口与神经模型构造）、`torch_models.py`（N-BEATSx、TFT 纯 PyTorch 实现）、`compact_neural.py`（早期紧凑实现） |
| `experiments/demand/` | 实验脚本，每个实验一个文件夹，均从仓库根目录以 `python -m experiments.demand.<文件夹>.<脚本>` 运行（见下表） |
| `results/demand/` | 汇总表、协议与 README；逐点预测（`*.csv.gz`）和模型检查点（`*.pt`、`*.joblib`）体积过大，不入库 |
| `submission/` | 提交包：需求部分为 N-BEATSx（最近四年重训）；价格部分暂为基线占位 |

六个需求模型：GAM、LightGBM、N-BEATSx、TFT、TCN、LSTM。选定模型为 N-BEATSx，最终策略为定期重训（最近四年数据）。

## experiments/demand/ 各文件夹

按时间顺序：

| 文件夹 | 实验 | 主要脚本 | 结果 |
|---|---|---|---|
| `pilot/` | 2014 初试（GAM） | `pilot_gam.py` | 终端输出 |
| `screen/` | 早期紧凑实现：2014 特征初筛与 2015–2026 冻结回测；防泄漏测试 | `run_demand.py`、`summarize_demand.py`、`test_forecast_boundaries.py` | `results/demand/screen/`、`frozen/` |
| `convergence/` | 神经模型收敛审查 | `check_convergence.py`、`summarize_convergence.py` | `results/demand/convergence*/` |
| `early_adaptation/` | 早期定期重训与 LightGBM 漂移检测 | `periodic_demand.py`、`adapt_demand.py`、`live_demand.py` | `results/demand/periodic/`、`adaptation/` |
| `config_search/` | Stage 1 正式配置搜索与多种子复核 | `tune_demand.py`、`test_formal_models.py` | `results/demand/formal/<模型>/`、`formal_gpu/` |
| `backtest/` | Stage 2 冻结与定期重训回测、汇总、种子敏感性 | `formal_backtest.py`、`report_formal.py`、`repeat_nbeats_backtest.py`、`check_lightgbm_seeds.py` | `results/demand/formal/backtest/`、`formal_report/`、`nbeats_seed_repeats/`、`lightgbm_seed_check/` |
| `error_analysis/` | 分组误差与 N-BEATSx/LightGBM 集成 | `analyze_demand_errors.py` | `results/demand/error_analysis/` |
| `holiday_ablation/` | 公众假日特征消融（exp0–exp3） | `holiday_ablation.py`、`holiday_ablation_neural.py` | `results/demand/holiday_ablation*/` |
| `drift_retrain/` | exp2 特征下的漂移触发重训 | `adapt_exp2.py` | `results/demand/formal_exp2/adaptation/` |
| `torch_rewrite/` | 纯 PyTorch 重写与 neuralforecast 逐位对比 | `check_torch_models.py` | 终端输出 |

TFT/TCN 的正式搜索在 UNSW Katana GPU 上运行；作业脚本含个人路径，未入库，运行环境记录在 `results/demand/formal_gpu/`。
