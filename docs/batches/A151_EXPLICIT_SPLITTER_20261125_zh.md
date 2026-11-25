# 显式 splitter 落地：按声明列切时间/实体，并限制为单一评价划分

2026-11-25。承接 `A151_ISOLATION_REFUSAL_20261125_zh.md` 的"仍未做"第 2 项。**0 API**。

## 1 实现了什么

`superagent/splits.py` 新增两个划分器，`grouped_partition` 按 `evaluation_protocol.kind` 分派：

| kind | 划分器 | 规则 |
|---|---|---|
| `grouped_stratified`（默认） | 原有分组分层 | 不变 |
| `temporal` | `temporal_partition` | 按声明列**排序**，前 `train_fraction` 行训练、其余留出；**不打乱** |
| `entity` | `entity_partition` | 按种子哈希给实体定序，整个实体一次性分到一侧；**同一实体绝不同时出现在两侧** |
| 其它 | — | **显式拒绝**（`Unsupported declared evaluation protocol`） |

声明方式：

```json
{"kind": "temporal", "column": "date", "train_fraction": 0.8}
{"kind": "entity",   "column": "user", "train_fraction": 0.8}
```

`column`（也接受 `time_column`/`entity_column`）必须是 `input_schema.columns` 里声明过的列，且特征矩阵的列数要与声明一致；`train_fraction` 必须在 `[0.5, 1)`。

## 2 三条不妥协的检查

1. **列必须存在**：`Protocol column 'missing' is not declared in input columns`；
2. **两侧类域必须一致**：不一致就拒绝评分（`does not keep the same classes on both sides`），不静默算一个不可比的分数；
3. **实体不得跨界**：划分后还会再验一次交集，非空即报错。

## 3 一个必须一并改的结构问题

第一次端到端跑时间划分时失败：

```
Registered confirmation repeats the same rows; no independent partition available
```

原因：`freeze_partitions` 用 seed、seed+101、seed+202 生成 primary 与两个确认划分，而**时间/实体划分与 seed 无关**，三次必然得到同一批行。

处理：新增 `declared_split_names(task)` —— 显式协议**只注册 primary 一个划分**，并把这一事实写进状态的 `registered_confirmation_splits`（不再硬编码两个确认划分）。

后果（如实记录）：
- `ab-v1` 正常跑，复核依据只有 primary；
- 依赖 `registered_confirmation_splits == list(CONFIRMATIONS)` 的 `ab-finalists` 会**明确拒绝**，而不是拿同一批行假装独立复核。

## 4 测试（10 项）

| 组 | 用例 |
|---|---|
| temporal | 顺序正确（训练侧最大日期 < 留出侧最小日期）、缺列拒绝、`train_fraction` 越界拒绝 |
| entity | 实体不跨界、同 seed 可复现、`entity_column` 别名可用 |
| 不支持 | 未知 kind 拒绝、协议非对象拒绝、默认协议仍正常分组 |
| **端到端** | 带 `temporal` 协议的包真实跑通，退出码 0，`primary_split.json` 的 `split_resource.protocol == "temporal"`，训练+留出 = 60 行 |

宽口径回归 **239 项通过**（27 个测试文件）。

## 5 端到端证据

包：60 行表格，列 `date,user,value`，声明 `{"kind":"temporal","column":"date","train_fraction":0.75}`。

运行结果：`status: completed`，`primary_split.json` 记录 `protocol: temporal`、`column: date`、`train_rows 45 / search_rows 15`，三次 fit 正常完成，留出侧由**最晚的 15 行**构成。

## 6 尚未做

1. **确认划分与显式协议的组合**：现在只注册 primary，`ab-finalists` 会拒绝。要支持"时间协议下的复核"，需要定义"用更晚一段做复核"或"多个时间折"，属新设计；
2. **实体协议的类域完整性**：实体分组可能天然导致某类只出现在一侧，现在直接拒绝；是否允许加权重采样或换种子重排，需要另定规则；
3. **真实任务迁移**：仍需另立数据资格/划分/预算计划。

## 7 资源

0 API、0 账本写入（端到端验证用本地决策，3 次物理 fit）。账本仍 457 请求 / known ¥7.925792 / unknown 16 / pending 0。
