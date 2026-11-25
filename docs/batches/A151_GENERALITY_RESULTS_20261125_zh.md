# 第四批结果：A 线通用性五个场景，四条通过、一条暴露口径问题

2026-11-25。按 `docs/NEXT_STAGE_TASK1_AND_INTEGRATION_20261125_zh.md` 第四批。**0 API**，共 15 次物理 fit（每个场景 3 次，只有 5 个场景中的 4 个真正训练）。

## 1 五个场景（匿名合成包，只验行为不验分数）

每个包都是 300 行单文本、manifest-v1、**文件名不含任何任务提示**。

| 场景 | 声明 | 结果 | 判定 |
|---|---|---|---|
| 非 accuracy 指标 | `metric: macro_f1` | 完成，`primary_metric=macro_f1`，任务契约与 trial 指标一致 | **通过** |
| 正类改变 A | `binary_f1` + `positive_label: yes` | 完成，`positive_label=yes` 被记录 | **通过** |
| 正类改变 B | `binary_f1` + `positive_label: no` | 完成，`positive_label=no` 被记录 | **通过** |
| 声明冲突 | `accuracy` + `positive_label: yes` | **照常运行**，`primary_metric=accuracy`，正类被记录但**未被使用、也未提示** | **暴露口径问题** |
| 资产缺失 | 声明 `train` 分块但包内没有该成员 | **拒绝**：`Missing or ambiguous split archive member`（退出码 2） | **通过** |

所有场景的验证分都是 1.0——合成数据线性可分，**分数没有信息**，这里只验证"行为是否正确"。

## 2 一处我自己差点写错的判定

我最初把"`roc_auc` + 三类标签"当成声明冲突，加了"roc_auc 只允许两类"的守卫。跑完既有测试才发现：**A 线本来就支持三类 roc_auc 的 macro one-vs-rest**（`test_multiclass_auc_is_macro_ovr_with_aligned_columns`），守卫是错的，已回退。

真正的冲突是上表第 4 条：**声明了正类、但主指标是 accuracy，正类不会进入任何计算**。系统照跑、把 `positive_label` 记成 `confirmed`，读报告的人会误以为它生效了。

## 3 与 B 的覆盖分工（不重复做）

| 场景 | B 已有 | 本轮 A 做了什么 |
|---|---|---|
| 声明冲突（Released 标签冲突） | `scripts/audit_b_released_label_conflicts.py` + 结果文档 | 换到匿名合成包的正类/指标冲突 |
| 复核重叠证据 | `scripts/audit_b_a29_confirmation_conflicts.py` | 第一批已用真实索引算过（重叠 18.2%–35.6%） |
| 可选资产 | `tests/test_unified_handoff_optional.py`（缺失时不接管） | **未做**（见第 6 节） |
| 匿名任务联合运行 | `scripts/run_a127_anonymous_joint.py` | 换成本轮四个行为场景 |
| 中断/恢复 | 无 | A150：中断 1 fit 后续跑，1+2=3 fit 与一次跑完一致 |

## 4 数值表格的负迁移检查

任务 `dry_bean`（数值表格）在 `full` 目录下只有 **5 个候选**，**不含 `supported_field_interactions`**：

```
numeric_tabular_logistic_C0.1 / C1.0 / C10.0 / extra_trees_leaf1 / extra_trees_leaf3
```

**结论：Task1 的交互路线不会泄漏到数值任务。** 这只是兼容性检查；数值表格不能替代"原生类别表格新任务"的迁移证据，计划里也已这样要求。

## 5 能说什么、不能说什么

**能说**
1. 匿名包上，非 accuracy 指标（macro_f1）从任务契约贯穿到 trial 与最终评分。
2. 正类的声明改变会被记录，两种取值可区分。
3. 声明的分块缺失时，入口**明确拒绝**而不是静默降级。
4. 交互路线不会出现在数值表格目录里。

**不能说**
1. **不能说通用性已证**：全是合成数据、单文本、300 行，验证分 1.0 无信息。
2. **不能说迁移成功**：没有跑任何一个真实新任务。
3. **不能说声明冲突被正确处理**：第 4 条恰恰说明它被**静默接受**。

## 6 还没做（明确）

1. **可选资产损坏**：没有构造"编码器权重文件存在但内容损坏"的用例；现有 B 的 optional 测试只覆盖缺失，不覆盖损坏。
2. **实体/时间隔离的声明式拒绝**：`grouped_partition` 对 `evaluation_protocol.kind != 'grouped_stratified'` 会明确拒绝（代码里已有），但**没有端到端用例**证明这条拒绝路径在真实入口上触发。
3. **真实任务迁移**：需另立计划（新数据资格/划分/预算），本轮不擅自下载。

## 7 资源

| 项 | 值 |
|---|---|
| API | **0** |
| 物理 fit | 15（4 个场景 × 3；缺失资产场景 0 次） |
| 费用 | ¥0 |
| 账本 | 未写，仍 457 请求 / known ¥7.925792 / unknown 16 |

## 8 建议的下一步（按计划第四批剩余项）

优先级从高到低：

1. **声明冲突的显式策略**：`positive_label` 在主指标不使用时应至少标成 `not_applicable` 或给出警告，而不是 `confirmed`——这是小的诚实性修复。
2. **可选资产损坏用例**：构造"权重存在但哈希不符"，验证入口拒绝。
3. **实体/时间隔离的端到端拒绝用例**：用带 `evaluation_protocol.kind='temporal'` 的包，验证入口确实拒绝而不是降级。
4. **真实迁移**：需先定数据资格与预算。

证据：`research/a151_generality/attempt_04/{packages,results.json,<case>/}`；`attempt_01/02/03` 为过程中的失败件（分别因目录已存在、输入列声明错误、冲突用例选错），均保留。
