# Unseen Task 就绪检查：能跑的形态已验四种，剩余风险三项

2026-11-25。用**交付入口**（`unified_agent`）实测四种可能出现的任务形态，并列出仍然存在的风险。**0 API。**

## 1 已验证能跑通的四种形态

脚本：`scripts/probe_unseen_shapes.py`（可重复执行，自带造包）。

| 形态 | 结果 |
|---|---|
| 表格 + **无标签 test 分块** | 9 fit、0 API、预测 60 行，`row_id,prediction` |
| **多分类** + `macro_f1` | 9 fit、选中 `open_QuadraticDiscriminantAnalysis_…`、预测 60 行 |
| 二分类 + `roc_auc` | 9 fit、预测带 `score_0,score_1` 两列 |
| **图片文件夹** + 独立特征包 | 10 fit、选中 `image_extra_trees_leaf1`、预测 200 行 |

结论：**只要封装格式落在我们支持的三种约定内，模态（表格/文本/图片）、类别数（二/多）、指标（accuracy / macro_f1 / roc_auc）、以及"只有训练标签、测试无标签"这几种变化都已经能端到端跑通。**

## 2 剩余风险（按严重程度）

### 风险一：格式约定之外一律拒绝（已有专门文档）

见 `UNSEEN_TASK_FORMAT_RISK_FOR_B_20261125_zh.md`。普通 CSV（任意列名、无声明）、Parquet、另一套图片目录结构 → 拒绝、零输出。

### 风险二：**指标声明不被识别时会静默退回 accuracy**

实测（`legacy` 旧格式包 + README）：

| README 原文 | 解析出的指标 | 来源 |
|---|---|---|
| `report macro F1` | **accuracy** | `legacy_adapter_default` |
| `report test accuracy` | accuracy | `declared_task_readme` |
| 什么都没写 | accuracy | `legacy_adapter_default` |

也就是说：**如果课程公布的指标是 F1/AUC 而包里的声明不是我们认的那几种措辞，我们会按 accuracy 选模**。`goal_resolution` 里会标 `unknown`（诚实），但**选模方向是错的**，外层按真实指标评分时会吃亏。

同一问题的另一面：manifest 里指标名是**大小写与连字符敏感**的：

```
F1 / f1 / macro-f1 / AUC / Accuracy  → 全部拒绝（Supported primary metrics are …）
macro_f1 / roc_auc / accuracy        → 接受
```

**建议**：① 指标名做规范化（小写、`-`→`_`、`f1`→`macro_f1` 这类显式映射）；② README 语法扩展到 `F1`/`AUC`/`precision`/`recall` 的常见措辞；③ 指标来源是 `unknown` 时，在报告与日志里**显著标注**，不让它悄悄当 accuracy。

### 风险三：运行失败 = 零输出

我们已经遇到过一次真实挂起（单次调用 417 秒后断流）。当前失败路径只写 `run_outcome.status=failed`，**不产出预测文件**。而课程 §5 明确把 "valid partial results from incomplete tasks" 计入分数。

**建议**：加"兜底交付"——契约无法建立或运行失败时，仍输出合法格式的预测（多数类），日志写明"保底、非方法结论"。

### 次要风险

| 项 | 说明 |
|---|---|
| 行号约定 | 我们的 `predict` 输出 `row_id = 0..N-1`；若课程要求原表行号或特定顺序，需要适配 |
| 本地规划的评分风险 | 交付是 0 API 本地规划；合规（课程限制的是"用哪个 LLM"），但 30% 的"自主开发"分项可能期待 LLM 参与规划，**这是呈现/报告风险，不是功能风险** |
| 大任务时长 | 24 小时窗口足够（表格 9 fit 约 11 分钟、图片 10 fit 十几分钟），但若数据规模大十倍需重估 |

## 3 结论

| 维度 | 现状 |
|---|---|
| 模态覆盖（表格/文本/图片） | **通过**（四种形态实测） |
| 类别数与指标多样性 | **通过**（多分类、macro_f1、roc_auc） |
| 无标签测试集 | **通过** |
| 封装格式覆盖 | **不足**（只有三种约定） |
| 指标声明识别 | **不足**（措辞不匹配会静默用 accuracy） |
| 失败兜底 | **缺失**（失败即零输出） |

**冻结前值得做的三件事（按性价比排序）**：① 兜底交付（最便宜、直接保分）；② 指标名规范化与 README 措辞扩展；③ 格式适配器扩展。

三件都是 0 API、可本地测试，不动现有默认行为。
