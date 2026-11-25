# 第四批补充：声明诚实性与资产损坏拒绝

2026-11-25。承接第四批报告第 8 节的两项建议。**0 API**，验证跑 3 次物理 fit（仅一次复核运行）。

## 1 修了什么

### 1.1 声明了但没被使用的正类，不再标成 `confirmed`

上一批发现：包声明 `positive_label: yes` 而主指标是 `accuracy` 时，正类被记成 `confirmed`，读报告的人会以为它影响评分。

现在 `resolve_goal` 会区分三种情况：

| 情况 | status | 附带信息 |
|---|---|---|
| 主指标使用正类（`binary_f1`/`roc_auc`）且声明了 | `confirmed` | — |
| 主指标**不使用**正类却声明了 | **`not_applicable`** | `reason`：该正类因主指标不针对单一正类而未被使用 |
| 未声明且主指标不使用 | `not_applicable` | 无 `reason` |

### 1.2 运行记录里补上 `goal_resolution`

更根本的问题：`task_spec.json` 原先只写 `task.spec` 原样字段，**manifest 包的运行记录里没有 `goal_resolution`**——也就是说，最需要「声明 vs 生效」溯源的匿名任务，恰恰没有这份记录。现在 `finalization` 在写 `task_spec.json` 时补上 `goal_resolution`（缺失时从 `TaskSpec.from_task` 取）。

## 2 端到端复核（3 fit，0 API）

对同一个「`accuracy` + `positive_label: yes`」的匿名包重跑一次：

```
status      completed
metric      accuracy
positive_label field:
  {"value": "yes", "source": "declared_spec", "status": "not_applicable",
   "reason": "the declared positive label is unused because the primary metric
              accuracy does not score a single positive class"}
```

**声明仍然完整保留**（value/source 都在），但状态不再谎称生效。证据：`research/a151_generality/attempt_06/conflict_recheck/task_spec.json`。

## 3 可选资产损坏：拒绝行为已被测试锁定

第四批报告里"可选资产损坏未做"这一项现在补上了，用的是入口自己的校验函数 `tools/encoder_assets.verified`：

| 用例 | 期望 | 结果 |
|---|---|---|
| 资产目录缺失 | 拒绝 | ✓ `ValueError` |
| 文件存在但内容不符（哈希不匹配） | 拒绝 | ✓ `ValueError` |
| 报错文本指名具体文件 | 是 | ✓ 错误信息含文件名 |

这三个用例证明：**损坏的权重不会被当成"缺失"或"可用"**，而是明确拒绝。

## 4 测试

新增 `tests/test_declared_assets_and_labels.py`，7 项：

- 正类被 `binary_f1` 使用 → `confirmed`；
- 正类被 `accuracy` 忽略 → `not_applicable` + reason；
- `macro_f1` 也不使用正类 → `not_applicable`；
- 未声明正类 → `not_applicable` 且无 reason；
- 资产缺失 / 资产损坏 / 错误指名文件。

宽口径回归 **210 项通过**（23 个测试文件，含 B 的 unified 与 portfolio 套件）。

## 5 仍未做

1. **实体/时间隔离的端到端拒绝用例**：代码路径存在（`evaluation_protocol.kind != 'grouped_stratified'` 即拒绝），但没有真实包证明它在入口上触发；
2. **真实任务迁移**：需另立数据资格/划分/预算。

## 6 资源

0 API；复核运行 3 次物理 fit；¥0。账本未写：仍 457 请求 / known ¥7.925792 / unknown 16 / pending 0。
