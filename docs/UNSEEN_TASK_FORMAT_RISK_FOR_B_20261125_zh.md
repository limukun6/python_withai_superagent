# 给队友的提醒：Unseen Task 的封装格式风险（冻结前必须处理）

2026-11-25。本文只讲一个问题：**如果 Unseen Task 的封装格式和我们已支持的三种不一致，提交的冻结版本会直接退出，产出为零。**

## 1 课程原文（关键三句）

来源：`D:/学校/python with ai/AIA3102 - Final Project - Full Assignment Requirements.md`

§2：
> Data formats, evaluation data splits, and metrics **will be announced with each task**.

§6：
> The Unseen Task must be completed using the SuperAgent version **frozen at Final Submission**. **No human intervention is allowed during execution.** Groups will have 24 hours to run their submitted SuperAgent and submit predictions and execution logs.

§5（评分表，两处都有这句）：
> including **valid partial results from incomplete tasks**

**三句连起来的意思**：格式在任务发布时才知道，但你只能用冻结版本跑，中途不能改代码。所以——

> **看到格式的那一刻，已经没有补救机会了。格式覆盖必须在冻结前做完。**

## 2 我们现在支持什么（三种，都有实证）

| 约定 | 识别依据 | 实证 |
|---|---|---|
| npz 旧格式 | 文件名 `C_train.npy` + `y_train.npy`，`Readme.md` 里出现 `report accuracy` | `released_task1.zip` 实测内容：`C_train.npy / C_val.npy / C_test.npy / y_train.npy / y_val.npy / y_test.npy / Readme.md / info.json` |
| 成对文本 CSV | 固定列名 `headline_A` / `headline_B` / `label_pairwise` | `released_task2.zip` 实测内容：`train.csv / val.csv / test.csv / README.md` |
| manifest-v1 | 包内 `manifest.json` 声明 `columns / target / metric / splits` | 我们自己的迁移任务 Car Evaluation 就是手写 manifest 才跑起来的 |

**再没有第四种。** 普通 CSV（任意列名、无 manifest）、Parquet、另一套图片目录结构，全部拒绝。

## 3 拒绝时会发生什么（实证）

以“声明了 `train` 分块但包内缺该成员”为例：

```
status: failed
error: Missing or ambiguous split archive member
attempted_classifier_fits: 0
partial_trial_models: []
delivery_status: unavailable
```

**0 fit、0 API、没有预测文件。** 对评分而言，这等于该任务 0 分（Unseen Task 占 10%）。

## 4 为什么这是真实风险，不是理论担忧

1. **格式是课程定的，不是我们定的**：released 两个包恰好是我们已支持的两种旧格式，不等于 unseen 包也是。
2. **冻结后不能改**：§6 明确禁止执行期人工干预，所以不能“看到格式再写适配器”。
3. **我们的 Agent 不会改自己的代码**：课程 §3 允许“generate or modify code”，但我们没有实现代码生成，只有注册好的方法库。因此它无法在运行时自适配新格式。

## 5 两件建议在冻结前做的事

### 5.1 拓宽格式覆盖（真正的修复）

按“**可核验的声明**”扩展适配器，优先级从高到低：

| 新增 | 识别条件 | 为什么不算猜 |
|---|---|---|
| 单表 CSV + 声明 | manifest 或 README 明确写“标签列是 X、指标是 Y” | 读的是公告，不是推测 |
| CSV + 独立 test 文件 | `train.csv` + `test.csv`，README 说明列含义 | 同上 |
| 图片文件夹 | 按类别名分子目录（`train/cat/…`、`train/dog/…`） | 目录结构本身就是声明 |
| 多分类表格 | 现有 tabular 路径已支持多类 | 已有 |

**不做**：按“最后一列是标签”“列名含 label 就猜”这类启发式硬接——那会把不确定装成确定。

### 5.2 兜底交付（保底分数）

当契约无法建立时，**仍然输出合法格式的预测文件**（例如多数类），日志里如实写：

> 任务契约未建立（原因：…）。本次输出为保底预测，不作为方法结论。

理由：§5 明确把 “valid partial results from incomplete tasks” 计入分数。现在“拒绝且无输出”是 0 分；有保底输出至少拿到部分分，且完全诚实。

**红线**：兜底必须显式标注，绝不能在报告里当成正式成绩。

## 6 分工建议

| 谁 | 做什么 |
|---|---|
| **A（我们）** | 适配器与格式识别、兜底交付路径、清单与日志口径、冻结前验收 |
| **B（队友）** | 确认方法库在新增格式（多分类表格、图片文件夹）下能正常注册与运行；确认没有依赖任务名的分支 |

## 7 验收标准（做完要能这样证明）

1. 用 4 种封装各造一个最小包（npz / 成对 CSV / manifest CSV / 图片文件夹），**每个都能跑通并产出预测**；
2. 用 2 种“声明缺失”的包验证**兜底交付**：退出前写出预测文件，`run_outcome` 明确标 `contract_unresolved=true`；
3. 任一任务名不出现在任何分支里（`grep` 任务名应为 0 命中）；
4. 全部 0 API，测试纳入回归套件。

## 8 需要队友回答的问题

1. B 的 `unified_agent` 是否已经支持“普通 CSV + 声明”或“图片文件夹”这两类封装？如果已支持，我们只需对接；
2. 方法库对**多分类表格**和**图片文件夹**是否都已注册（当前 Task1 是多分类表格、Task2 是成对文本，图片只在 A88 做过）；
3. 冻结时间点定在哪天？决定这两件事有多少时间窗口。

## 9 现状一句话

**我们已经把“不确定就拒绝”做得很干净，但课程给的是“格式公布后不能改代码”，所以“拒绝”在这里的代价不是一次实验失败，而是直接丢掉一部分分数。** 冻结前把格式覆盖和兜底交付补上，是把“可能 0 分”降成“最差也有保底”的唯一办法。
