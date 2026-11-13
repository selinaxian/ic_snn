# IC标准分钟行情 v1

逐日数据已按序列合并并排序，原始行情数值保持不变。具体合约与滚动序列分开保存。

- 输入：191个原始压缩包；ZIP/7z按文件头识别。
- 具体合约：7个，合计124,800行。
- 滚动序列：6个，合计267,840行。
- 隔离行：0；删除的完全重复行：0。所有处理均保留来源。
- 供应商是否调整价格、分钟开始/结束标签和量单位仍未知；没有自行复权、填补或猜定合约。

| 序列 | 类型 | 首日 | 末日 | 有数据天数 | 行数 |
| --- | --- | --- | --- | ---: | ---: |
| IC2606 | contract | 2025-12-04 | 2026-06-18 | 129 | 30,960 |
| IC2607 | contract | 2026-05-18 | 2026-07-17 | 44 | 10,560 |
| IC2608 | contract | 2026-06-23 | 2026-08-21 | 40 | 9,600 |
| IC2609 | contract | 2026-01-19 | 2026-09-15 | 156 | 37,440 |
| IC2610 | contract | 2026-08-24 | 2026-09-15 | 16 | 3,840 |
| IC2612 | contract | 2026-04-20 | 2026-09-15 | 98 | 23,520 |
| IC2703 | contract | 2026-07-20 | 2026-09-15 | 37 | 8,880 |
| ICL0 | continuous | 2025-12-04 | 2026-09-15 | 186 | 44,640 |
| ICL1 | continuous | 2025-12-04 | 2026-09-15 | 186 | 44,640 |
| ICL2 | continuous | 2025-12-04 | 2026-09-15 | 186 | 44,640 |
| ICL3 | continuous | 2025-12-04 | 2026-09-15 | 186 | 44,640 |
| ICL8 | continuous | 2025-12-04 | 2026-09-15 | 186 | 44,640 |
| ICL9 | continuous | 2025-12-04 | 2026-09-15 | 186 | 44,640 |

## 读取

```python
import pandas as pd

# 同一个具体合约的全部日期
bars = pd.read_parquet("data/standardized/ic_v1/contracts/IC2606.parquet")

# 全部具体合约；同一分钟存在多个合约是正常的
all_contracts = pd.read_parquet("data/standardized/ic_v1/contracts")
assert not all_contracts.duplicated(["contract_id", "datetime"]).any()

# 只取所需日期，再按合约分别计算特征
part = all_contracts[all_contracts.trading_date.between("2026-01-01", "2026-03-31")]

# 通过source_id定位原始文件
sources = pd.read_csv("data/standardized/ic_v1/metadata/source_members.csv")
row = bars.iloc[0]
origin = sources[sources.source_id == row.source_id].iloc[0]
print(origin.source_archive, origin.source_member, row.source_row)
```

`source_archive`相对路径以manifest中的source_root为根；source_row为原CSV行号，表头通常是第1行。
`previews/`内是每个序列首尾各10行CSV，便于直接打开，完整数据在Parquet中。
价格为float64，成交量/持仓量为int64，datetime带Asia/Shanghai时区。
只有具体合约文件的contract_id有值；ICL0/1/2/3/8/9的contract_id保持空值。

## 质量与信息边界

参考时段是现有数据观察到的09:31—11:30、13:01—15:00，每日240分钟，尚未由供应商确认时间标签定义。
`metadata/daily_quality.csv`逐序列逐包日期区分未提供文件、空文件、缺分钟等；只覆盖本批实际存在的包日期，不等价于已核查完整交易所日历。
`metadata/gaps_within_observed_span.csv`列出该序列首次至末次有行情之间的参考网格缺失，不把生命周期外日期直接认定为数据丢失。
`metadata/quarantine.csv`保存不合法行和同键冲突；`duplicates.csv`保存全部重复来源与处理决定。零成交量为提示，不自行替换行情。
全部价格、成交量和持仓量都来自对应文件。ICL8与IC2606的6月18日差异仍各自保留，未合并成一套价格。
复制的`audit_*`表仅在相关源文件哈希与之前审计完全相符时保留，表示事后匹配证据，不是官方主连规则或交易时点可用的主力选择信号。

后续特征、未来收益和神经状态必须按合约/连续交易时段分组；所有合约使用统一训练/验证/测试日期边界。
本次没有改动原final.csv、旧模型或训练程序；现有训练入口仍读取CSV，使用本库训练需要单独接入数据读取和分组逻辑。

## 来源与复现

`metadata/archives.csv`保存原包路径、真实格式、大小、mtime及SHA256；`source_members.csv`保存每个成员的内容哈希和来源索引。
`metadata/schema.json`记录字段和未知口径；`manifest.json`记录配置、代码/产物哈希及依赖版本；`requirements-lock.txt`保存本次运行环境。
源文件内容与来源核验结果记录在manifest和audit evidence中。原始包及final.csv未覆盖。

在项目根目录运行：

```bash
.venv/bin/python -m pip install -r requirements-data.txt
.venv/bin/python -m snn_data.standardize --source data/分钟线 --output data/standardized/ic_v2
```

输出目录必须不存在，以免覆盖既有版本。修改源数据或规则后生成新版本。
