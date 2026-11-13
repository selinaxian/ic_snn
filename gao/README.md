# gao：多因子输入的 Gao 风格事件交易基线

独立项目。12因子/相反数→H1/H2各64个LIF→一个输出LIF。保留Poisson编码，两层连接均STDP学习；输出事件决定是否交易，动量、Alexander Filter、随机振荡器分别决定多空。没有岭回归、deSNN、储备池或隐藏层赢家竞争。

**完整180对日期基线已完成。** 结果见[完整报告](results/baseline/REPORT.md)。当前STDP触发率92%–93%，未见有效事件选时增量，成本情景明显亏损；没有根据结果修改基线。工程核验见results/baseline/verification.json。

**这是用户授权的多因子、分钟数据适配，不是原论文全部软件和收益的精确复现。** 原文未公布的参数固定写入config.json，来源和适配见[PROTOCOL.md](PROTOCOL.md)。论文副本[docs/Gao_2021.pdf](docs/Gao_2021.pdf)。

## 结构

| 文件 | 职责 |
|---|---|
| src/data.py | 独立行情读取、12因子重算、分钟/合约/半日边界验证 |
| src/encoding.py | 训练日正均值平移标准化、正反双分支Poisson编码 |
| src/network.py | 两分支LIF、所有连接STDP、冻结、状态重置、批量独立日期仿真 |
| src/strategy.py | 三种方向规则、论文式逐笔账本、成本情景、等次数随机时点 |
| src/evaluation.py | 论文式事件标签和指标；标签不供训练交易 |
| run.py | 前日训练/次日测试、三个固定种子、对照、模型/信号/策略落盘 |
| config.json | 全部固定数值假设 |
| tests/test_pipeline.py | 机制与成交边界检查 |

在本目录运行：

```powershell
python -m unittest discover -s tests -v
python run.py --output results/baseline
python verify_results.py --output results/baseline
python summarize.py
```

输出目录必须不存在，防止覆盖。默认完整运行180对日期、三个种子、STDP及同初始化随机网络。`--days 2`仅用于工程冒烟检查，不能作为完整结果。

主要输出：minute_signals.csv（所有可输入分钟含无事件分钟）、event_metrics.csv、network_diagnostics.csv、paper_style_trades.csv、random_timing_reference.csv、cost_scenario_trades.csv、cost_scenario_daily.csv、strategy_metrics.csv、models/、traces/、date_splits.csv、source_hashes.json、complete.json。论文式重叠交易收益和不可叠仓成本情景分别报告。
