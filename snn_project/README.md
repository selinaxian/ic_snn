# IC 分钟线 SNN 基准

最新完整响应诊断：`experiments/response_value_audit/REPORT.md`。复用三个历史折冻结模型，对真实编码输入做时点扰动，并比较原始因子、原始因子＋随机池、原始因子＋STDP 池对下一分钟绝对收益的预测。保存 13,800 行分钟输出，当前没有 STDP 增量。入口 `scripts/response_value_audit.py`；独立复算 `scripts/verify_response_audit.py`。

从 snn_project 目录运行。先执行 pip install -r requirements.txt。原始数据见上一级 data 目录，项目输入和审计记录在 data，历史预测、指标和权重在 experiments。

```powershell
python 'scripts/run_baseline.py' --output 'experiments/new_run'
```

输出目录须是新目录。metrics.json 记录训练、验证和测试日期、指标及储备池诊断，predictions.csv 保存验证与测试预测，NPZ 保存权重、输入映射、标准化参数和训练期近邻样本。


## 代码结构

实验入口在 scripts，按数据流拆分的组件在 scripts/snn_core。

| 组件 | 职责 |
|---|---|
| market_data.py | 读取分钟 CSV，校验时间顺序，按连续时段分段，生成 OHLC、量和仓位衍生特征 |
| encoding.py | 主线确定性因子阈值编码、Gao 式速率编码、论文 TR 编码 |
| snncube.py | 三维储备池拓扑、输入映射、膜电位推进、记录发放 |
| stdp.py | 用发放先后更新储备池的兴奋性突触 |
| spike_features.py | 主线的分段发放率 |
| desnn.py | deSNN 发放签名和近邻预测 |
| ridge_readout.py | 拟合训练集标准化和岭回归读出 |
| metrics.py | Rank IC、RMSE、方向匹配率、按交易日配对重采样 |

主线入口为 scripts/run_segmented.py。实验切分、样本对齐和输出由入口脚本编排。market_data.py 从整理好的 CSV 开始，原始归档的提取与清洗记录见 data 下的审计文件。
## 实验协议

- data/final.csv 是 2026 年上半年的 ICL8 中证 500 主连分钟线。按连续分钟分段，午休、隔夜和缺口处重置仿真状态，窗口和标签均在段内。主连切换及实际合约成交信息尚待数据源确认。
- OHLC 组用收盘对数收益、K 线实体、振幅和开盘相对前收盘的变化。扩展组加成交量和持仓量的 log1p 差分。每个变量及其相反数各占一路。
- Gao 式非负速率编码。训练段统计量用于标准化，正偏移后截断到 [0,1]。每分钟用 8 个 Bernoulli 小步近似离散 Poisson 发放。随机种子为 42，同一输入组的随机储备池与 STDP 储备池使用相同脉冲和初始连接。
- NeuCube 风格的 4×4×4 LIF 循环储备池，按距离抽取稀疏连接，输入端口随机固定。训练期连续推进，每段只学习一遍。STDP 更新兴奋性突触，抑制性突触固定。训练完成后冻结全部突触，用同一套权重重新仿真提取活动。
- 默认每分钟采集过去 30 分钟的发放记录。deSNN 风格的窗口签名由每个神经元的首次发放排名与后续发放漂移组成；使用 3 近邻标签均值读出未来 5 分钟的收盘到收盘对数收益，单位 bp。这是轻量近似，输入映射、STDP 实现和读出细节与论文软件有差异。
- 按日期切分 81/17/18 个交易日。对照包括预测零收益、相同历史上的原始特征近邻、相同初始化的冻结储备池近邻。指标是 RMSE、Rank IC 和符号匹配率。预测零收益的符号匹配率只是实际零收益占比。
- results 和 results_gain25 为早期每隔 5 分钟取样的运行，分别使用 --gain 1 和 --gain 2.5。results_minute 使用 --stride 1 和 --gain 2.5，相邻标签共享 4 分钟收益区间。增强连接是在首轮结果可见后进行的诊断，因此两轮均属于探索结果。当前是预测实验，未把主连预测换算成可执行交易或扣费收益。


## 储备池探索

`--side 4` 和 `--side 6` 分别使用 64 和 216 个神经元，连接抽样的期望平均入度保持相近，实际入度写入指标。`--depression` 是 STDP 抑制更新幅度与增强幅度之比。`--skip-knn` 只跑岭回归，省去高维近邻开销。

岭回归对照用训练集拟合的特征均值和标准差，正则系数固定为 0.1（均方损失口径）。比较原始滞后特征、deSNN 风格签名、窗口发放率。每种储备池特征都用随机冻结与 STDP 成对比较。所有预测仍逐分钟生成。

## 7–9 月留出检验

`holdout_jul_sep` 固定上一轮候选（216 神经元、种子 43、STDP 抑制比 1.0），保留原训练截止日 2026-05-11。5 月 12 日至 6 月 30 日为已见对照段，6 月 30 日之后为本轮独立留出段。模型逐分钟预测未来 5 分钟收益，比较原始因子岭回归、原始因子加随机池签名、原始因子加 STDP 池签名。三个模型都使用固定正则系数。

新输入表 ic_holdout_input.csv 由 data/final.csv 和 7 月 1 日至 9 月 15 日可读取、具有唯一 ICL8 CSV 成员且通过日期与 240 行检查的日归档合并而成。跳过项见 holdout_input_audit.json。实际样本起止以 metrics.json 为准。

```powershell
python 'scripts/run_baseline.py' --holdout --data 'data/ic_holdout_input.csv' --side 6 --depression 1 --seed 43 --skip-knn --output 'experiments/new_holdout'
```

## 论文 TR 与 deSNN 流程复刻

`paper_replication.py` 使用训练段相邻变化标准差的 0.5 倍作各通道阈值，正负事件分开输入。每个观测对应一个仿真步。对同一份发放记录，用首次发放排名与后续漂移构造 deSNN 向量，分别测试 mod=0.5 和 0.8。近邻版本用欧氏距离选 3 个邻居，再用 sigma=1 的高斯权重汇总；原文没有给出足够细节确认该核公式。岭回归使用相同向量。储备池沿用本项目的距离稀疏连接和 STDP 代码。

- QQQX 用 NeuCube-Py 附带的 50 个按顺序滑动的股价样本，截取每个窗口最后 60 步，前 25 训练、后 25 测试。前 49 个标签与下一窗口末尾的 QQQX 值逐一吻合。仓库未标记这些价格是否复权。基准为延用当前价格。Wipro 与新闻原实验未复现。
- IC 使用 60 分钟发放窗口、逐分钟预测未来 5 分钟收益。采用与前一轮相同的训练和留出日期。由于窗口从 30 改为 60 分钟，可比读出应在本轮内部比较。基准为零收益。
- 输出见 paper_tr_desnn。CSV 的 split 1 和 2 分别代表验证和测试。QQQX 的 RMSE 单位为价格，IC 为 bp。

```powershell
python 'scripts/paper_replication.py' --output 'experiments/new_replication' --ic 'data/ic_holdout_input.csv' --share-price '../04_项目代码/NeuCube-Py/example_data/share_price'
```

## 主线基准：确定性编码和分段发放率

`run_segmented.py` 使用 OHLC 衍生特征、成交量变化和持仓量变化。带方向的特征用正负双通道，振幅用单侧异常通道，共 11 路；阈值为训练段标准差的 0.5 倍。每分钟一仿真步，STDP 时间常数为 2 步。随机池和 STDP 池均为同初始化的 216 神经元储备池，后者只在训练期更新。连续时段内逐分钟推进，午休和隔夜重置状态。

三组岭回归都使用过去 30 分钟的连续因子。两组储备池分别补充最近 5、之前 10 和最早 15 分钟的发放率，共 648 维。预测目标为同段未来 5 分钟收益，按交易日成对重采样计算 Rank IC 差值区间。7–9 月为已查看过的诊断期，不能充当最终留出集。

```powershell
python 'scripts/run_segmented.py' --data 'data/ic_holdout_input.csv' --output 'experiments/new_segmented'
```

本次输出在 experiments/segmented_v1。



## 标签与发放顺序诊断

```powershell
python scripts/run_diagnostics.py --output experiments/new_segmented_diagnostics
```

使用 segmented_v1 的已保存储备池权重。先复核带符号收益的三组预测，再用相同输入预测绝对收益。顺序诊断仅对 STDP 池，比较原始因子、分段发放率再加真实逐分钟记录，与把同一分钟内所有神经元作为整体进行段内置换的对照。置换种子固定为 43，每窗口独立，逐段验证每个神经元发放计数相同。两个高维读出均以 float32 计算并沿用同一岭回归惩罚。按交易日成对重采样。5–6 月与 7–9 月均已用于历史探索。

## deSNN 近邻主线

```powershell
python scripts/run_desnn_neighbors.py --output experiments/new_desnn_neighbors
```

沿用 segmented_v1 的编码、已保存储备池、30 分钟窗口、逐分钟预测点与未来 5 分钟收益标签。三组用相同的欧氏距离 3 近邻与高斯加权规则，匹配空间分别为训练期标准化的连续因子、随机池 deSNN 签名和 STDP 池 deSNN 签名。脉冲签名使用 mod=0.5、drift=0.005，保持原有尺度。近邻库只含训练日期。输出含 Rank IC、近邻标签离散度、同号比例、与目标收益的误差和近邻涉及的交易日数。7–9 月是已查看过的诊断期。本轮结果在 experiments/desnn_neighbors_v1。

## 发放轨迹距离对照

```powershell
python scripts/compare_spike_distances.py --output experiments/new_spike_distance_comparison
```

固定既有 STDP 池、30 分钟窗口和三个近邻等权读出，对比保存的 deSNN 邻居、精确发放 Jaccard、沿时间轴用 [0.25, 0.5, 0.25] 平滑后的欧氏距离。边界外按零处理。每对邻居都记录三种距离、精确重合率和未来收益。图中四个面板使用同一神经元顺序和时间坐标。

## 无标签发放模式发现

```powershell
python scripts/discover_spike_patterns.py --output experiments/new_discovered_patterns
```

使用现有冻结的 STDP 和随机池。同一交易时段内取 30 分钟发放图。每隔五个训练窗口取原型候选，支持度和后续匹配均使用完整的逐分钟窗口。固定 Jaccard 下限 0.40，至少五个不同训练日支持，最多六个原型。不按成交量事件或钟点筛选。只匹配最接近且过线的原型。冻结原型后才关联行情和收益。收益摘要每模式每交易日只取首次触发，同钟点的全体窗口均值只作描述对照。图挑选各段内最相似的案例，按发放距离选择。既有日期都属于探索样本。

## 多脉冲父子扩展

```powershell
python scripts/expand_spike_patterns.py --output experiments/new_spade_pattern_expansion
```

沿用冻结储备池、五步模式范围与训练期至少 20 次的门槛。每个间隔从最常见的 24 个双脉冲组合出发，每轮仅保留训练出现次数最多的 24 个新组合；每池最多展开 240 个父分支，模式大小上限 216。搜索不读收益。若添加发放点保留完全相同的训练触发集合，则可合并为闭合扩展。本轮无此类跳跃。

输出含搜索进度、父子组合、各段触发保留率及对齐的收益比较。对同一个父模式，把是否满足子模式的事件都对齐到子模式最后一个脉冲应出现的分钟，再计算未来五分钟收益。每天保留全部分钟触发，区间按交易日成对重采样 400 次，未做多重比较校正。5–9 月是已用过的探索数据。结果位于 experiments/spade_pattern_expansion。

## 1 分钟读出的三轮历史复盘

```powershell
python scripts/walk_forward_spike_readout.py --output experiments/new_walk_forward_1m
```

训练段从 81 天扩展到 101 和 121 天，每轮接 20 天选型、20 天冻结测试。两池用同样的无标签模式发现流程，固定只读刚完成的位置。选型期至少出现 20 次且涉及至少 5 天的模式，按平均 1 分钟带符号收益的绝对值取前四个。读出是对应的选型期均值。多个模式同时命中取均值，未命中输出零。测试收益是分钟收盘到下一分钟收盘，按预测方向对齐的触发均值未计成本。三个测试段相互不重叠，已有数据属于历史复盘。结果见 experiments/walk_forward_1m_v1。

## 全窗口熟悉度与模式库审计

```powershell
python scripts/walk_forward_spike_readout.py --output experiments/new_library_audit
python scripts/whole_window_familiarity.py --output experiments/new_whole_window_familiarity
```

滚动读出脚本额外输出所有符合选型门槛的模式，不改变原来前四的预测。walk_forward_library_audit/library_summary.csv 比较全库复现和选型至测试的方向一致性，模式之间可能共享发放点与标签。

全窗口熟悉度把 5×216 发放图展平，在训练库里用二值 Pearson 相似度找第一近邻。该相似度根据两张图的总发放比例修正共同发放量。同钟点不限制搜索。取最近邻相似度在随后 20 天校准分布里的经验百分位，得到 0 到 1 的熟悉度。分数和参考库均不使用未来收益。每轮测试输出每分钟得分、匹配历史时刻、发放率、近期振幅和独立的事后标签。每连续时段最后一个可出分分钟也保留，其未来一分钟标签为空。

whole_window_familiarity/score_profiles.csv 用校准分布的 20% 和 80% 作为分组线，组内样本比例可随时间改变。未来绝对收益和同钟点均值仅作描述，不是得分输入。随机池熟悉度与近五分钟振幅百分位是两条参照。全部时段已用于历史研究。

## 熟悉度条件对照与循环活动诊断

`scripts/diagnose_window_familiarity.py` 复用原 13,800 行分数，核验行情与标签，诊断钟点、近期振幅、两池发放密度之外的关系。`scripts/audit_reservoir_recurrence.py` 按原参数重建并核验冻结记录，检查三步重复，并做前 15 步后停止外部输入的机制对照。未改模型参数或收益期限。

结果与完整限制见 `experiments/familiarity_conditional_audit/REPORT.md`。STDP 池有强三步重复，停止外部输入后仍维持高发放；熟悉度对下一分钟绝对收益的条件关系仍不稳定。此结论不等于完整响应没有市场信息。`diagnostic_residuals_NOT_LIVE.csv` 使用历史诊断性交叉拟合，不能当作可实时复算的策略信号。
