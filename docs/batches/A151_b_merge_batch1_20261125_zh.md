# 第一批结果：B 统一入口已适配到 A 线，实验有效性四项核对完成

2026-11-25。按 `docs/NEXT_STAGE_TASK1_AND_INTEGRATION_20261125_zh.md` 第一批执行。**0 API、0 新训练、未写账本。**

## 1 整合：合并到独立分支 `a151-b-merge`

| 项 | 值 |
|---|---|
| 基线 | A 线 HEAD `5920c23` |
| 合入 | `origin/codex/unified-released-1124`（`bcea278`） |
| 结果 | **`2618517`**（327 个文件变更，84,224 行新增） |
| 冲突 | 3 个：`superagent/ab_run.py`、`superagent/cli.py`、`docs/EXPERIMENT_PROGRESS.md` |

冲突处理原则：**合并两侧，不覆盖**。

| 冲突点 | 解决方式 |
|---|---|
| `ab_run.py` 决策分支 | 保留 A 的片段覆盖块，同时采用 B 的 `on_explore=choose if picker and not method_search else None` |
| `cli.py` 参数表 | 保留 A 的 `--llm-request-limit`、`--resume-from`、`--plan-fragment`、`--plan-fragment-file`，同时采用 B 的 `preference-catalog` 新选项 `varied` |
| `EXPERIMENT_PROGRESS.md` | 两侧段落都保留（A 在上、B 在下） |

B 带来的生产文件共 46 个，主要是 `unified_agent.py`、`superagent/unified.py`、`superagent/open_methods.py`、`superagent/ab_method_search.py`、`superagent/ab_varied_preferences.py`、`superagent/runtime_inventory.py`、`tools/*`（8 个）与 `scripts/*`（27 个）。**没有合并任何模型、账本或 `research/` 大目录。**

## 2 唯一需要改 B 代码的地方：编码

B 的证据助手 `scripts/unified_released_study.py::read` 用 `Path.read_text()`（系统默认编码）读自己用 UTF-8 写出的 JSON。本机默认编码是 **cp936**，于是 `D:\...` 路径被解成非法转义，B 的一个测试直接报 `Invalid \escape`。

修复：读取显式 `encoding='utf-8'`（1 个文件、1 处）。修复后 B 的 3 个该测试全过。

这是**平台可移植性缺陷**，不是合并引入的：在 UTF-8 机器上不会暴露。

## 3 实验有效性核对（四项）

### 3.1 复核记录里的"independent"不等于评价行无交集

从 44 个同时保存 primary 与 confirmation_1 划分索引的运行里直接算集合交：

| 项 | 值 |
|---|---|
| confirmation 的 search 行与 primary 的 search 行重合比例 | **最小 18.2%、均值 23.9%、最大 35.6%** |
| 含义 | 两个划分是**不同的随机划分**，但**评价行有实质重合** |

**结论**：`independent_confirmation_observed=true` 只能读作"换了另一个划分"，**不能读作"独立样本复核"**。B 文档里"独立进程/独立划分"的表述需要在我们的报告里改成"不同划分（评价行重合约 24%）"。证据：`research/a151_b_merge/split_overlap.json`。

### 3.2 指标来源与类序

| 任务 | kind | 指标 | 来源 | 状态 |
|---|---|---|---|---|
| `released_task1` | categorical_tabular | accuracy | `declared_task_readme` | confirmed |
| `released_task2` | pair_text | accuracy | `declared_task_readme` | confirmed |

两个任务的 `goal_resolution` 都把它标成 `confirmed` 且 `effective_policy=maximize`，成对文本的 `pair_semantics=preference` 来自 `input_schema`。**指标优先级符合"README 声明优先"**。

### 3.3 Task1 目录能否触达 B 的最优方法

当前 `catalog_scope=full` 为 `released_task1` 生成 **15 个候选**，其中：

- `catboost_depth8_reg3.0`：`depth=8, l2_leaf_reg=3.0, iterations=600, learning_rate=0.05`，**1 次 fit**；
- 另有 `catboost_depth8_reg10.0`、`catboost_depth4/6` 各两档，以及 `supported_field_interactions`。

**结论：B 的 Task1 最优方法在目录里、可被调度选中、单次 fit 成本为 1。** 但"9 fit 预算内是否真的选中它"属于第二批，本轮不下结论。

### 3.4 物理 fit 计数：1790 没有日志支撑

| 口径 | 数值 | 说明 |
|---|---:|---|
| 运行级 `classifier_fits.json` | **1107** | 155 条记录 |
| 批次级 `completion.json` | **1433** | 246 个批次完成件 |
| 当前 `CURRENT_AGENT_STATE` 顶部 | 1790 | **无对应日志** |

两个口径不等，是因为部分批次的 fit 没有落在运行目录里（或使用了不同的记录方式），而部分运行目录可能是同一实验的副本（内容完全相同的组有 7 组、109 次 fit，无法从内容区分"同一次实验复制"与"两次实验恰好拟合了相同候选"）。

**处理**：不写新的全局数字。在 `CURRENT_AGENT_STATE` 顶部改成"运行级日志已确认 ≥1107 次物理 fit；批次声明合计 1433；此前的 1790 无日志支撑"。

**同时纠正**：A149/A150 报告里"0 fit"的说法不成立——它们确实没有 API 调用，但**训练了**：A150 三个本地运行 3+1+3=7 次，A149 四组对照 4×3=12 次，合计 **19 次物理 fit**。"0 API"≠"0 训练"。

## 4 回归

合并树上 **165 项测试通过**（A 的 12 个测试文件 + B 的 5 个）。B 的 `test_unified_handoff_optional` 编码修复后通过。

## 5 本轮没做的

1. **训练读取白名单**未逐行核：只确认了入口只用训练成员与说明（B 文档声明），没有做"故意放入测试标签文件看是否被读"的负向测试。
2. **每次 trial 指标贯穿**未核：没有逐 trial 验证 `metric_spec` 与最终评分指标一致。
3. **未合并到 `codex/expanded-feedback`**：目前只在 `a151-b-merge`。是否并以何时并，取决于第二批是否从这个 SHA 起跑。
4. **未跑 B 的 `unified_agent` 端到端**：没有用原始包冷启动（需要训练，属第二批另登记）。

## 6 冻结信息

| 项 | 值 |
|---|---|
| 适配分支 | `a151-b-merge` |
| 生产 SHA | **`2618517`** |
| 基线 SHA | `5920c23`（A）/ `bcea278`（B） |
| 账本 | **未写**（仍 457 请求 / known ¥7.925792 / unknown 16 / pending 0） |
