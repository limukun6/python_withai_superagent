# 决策可复现性审计：48 次状态重放，以及一个需要撤回的结论

2026-11-20，面向 Codex 复核。本文只报告本机实测；所有数字都能用第 8 节命令复算。

## 0 一句话

`checked_loop_pilot` 记录的"检查组比原始历史组在文本上高 1.6579 个百分点"**不能作为结论**。把当时的状态逐条重放后，同一状态下两种上下文改变下一动作的比例（7/12）不高于无关数值噪声造成的改变比例（3/4）。

## 1 这次做了什么

| 项 | 值 |
|---|---|
| 输入 | `runs/checked_loop_pilot_v1`（11:55 完成，12 次 Flash 调用，66 拟合，¥0.086265） |
| 方法 | 从每条轨迹的 `decision_states.json` 重建当时的状态，用**生产代码里的同一个** `decision_context` 生成两种上下文，再问一次模型 |
| 规模 | 12 个状态 × 2 上下文 × 2 次 = 48 次调用，0 拟合 |
| 费用 | **¥0.371774**；另有一次失败尝试 `runs/replay_v1`（2 次调用 ¥0.010751）。本轮合计 ¥0.382525 |
| 耗时 | 13 分 50 秒（13:59:43–14:13:33 本地），单次响应 0.46–67.99 秒 |
| 产物 | `runs/replay_v2/{replay.json,completion.json,llm_usage.jsonl,events.jsonl}` |
| 工具 | `scripts/replay_same_history.py`（新增，未提交） |

模型、端点、温度（0）、提示词、候选菜单、`max_tokens` 全部沿用原实验；每条请求都是原实验逐字节的输入，见第 2 节。

## 2 前置校验：重放就是历史 prompt

`replay_same_history.py --verify` 用 `CheckedHistoryPolicy.decision_context` 从 `decision_states.json` 重建 12 条历史请求，与 `events.jsonl` 里实际发出的 payload 逐条比对：

```
request 0 task1/raw/d1: identical (5678 bytes)
...
request 11 task2/raw/d3: identical (14984 bytes)
```

**12/12 字节级一致**，零费用。所以第 3 节比的是"同一道题问两次"，不是"两道相似题"。

需要注意索引约定：`decision_states.json` 的第 0 条是共享 anchor（试验 1，本地决策，无调用），第 k 条对应第 k+1 次试验，也就是第 k 次模型请求。我第一版把 `--run` 写成 `[step-1]`，于是问了空历史状态，两次调用各 ¥0.0054 后按"失败即停"终止，产物留在 `runs/replay_v1`（未删，作为失败记录）。

## 3 三个观测

### 3.1 网关对同一请求复用响应，重复同 prompt 不是独立样本

48 次调用只有 **24 个不同的 `response_id`**，且每个 id 恰好出现 2 次，成对相邻；两两的 `input_tokens`、`output_tokens`、`course_cost_rmb` 完全相同。

```
('raw_history',     '33efa6', 1794, 1104)
('raw_history',     '33efa6', 1794, 1104)   <- 同一 id、同一用量、同一费用
('checked_history', '1172d0', 2091,  894)
('checked_history', '1172d0', 2091,  894)
```

也就是说：**"同一 prompt 重问两次答案总是一致（12/12）"不是模型确定性，而是网关把同一次生成回了两次，并且照常计费。**

这条直接影响以后所有"重复几次取多数"的设计：不改变请求内容、不换 cache 键，重复抽样拿不到独立样本。

### 3.2 跨时段、逐字节相同的 prompt，12 组里 3 组改了答案

拿 11:55 的历史回答与 14:00 的重放对比（同一手臂、同一状态、同一上下文）：

| 状态 | 11:55 历史答案 | 14:00 重放 | |
|---|---|---|---|
| task1 raw d1 | `catboost_depth4_reg3.0` | `catboost_depth6_reg3.0` | 变 |
| task1 raw d2 | `catboost_depth6_reg10.0` | `catboost_depth6_reg10.0` | 同 |
| task1 raw d3 | `catboost_depth8_reg10.0` | `catboost_depth8_reg10.0` | 同 |
| task1 checked d1 | `catboost_depth4_reg3.0` | `catboost_depth4_reg3.0` | 同 |
| task1 checked d2 | `catboost_depth4_reg10.0` | `catboost_depth6_reg10.0` | 变 |
| task1 checked d3 | `catboost_depth6_reg10.0` | `catboost_depth6_reg10.0` | 同 |
| task2 checked d1 | `word_char_C0.3` | `word_char_C0.1` | 变 |
| task2 checked d2 | `word_char_C0.1` | `word_char_C0.1` | 同 |
| task2 checked d3 | `word_char_stats_C0.3` | `word_char_stats_C0.3` | 同 |
| task2 raw d1 | `word_char_C0.1` | `word_char_C0.1` | 同 |
| task2 raw d2 | `word_char_C0.05` | `word_char_C0.05` | 同 |
| task2 raw d3 | `word_char_stats_C0.1` | `word_char_stats_C0.1` | 同 |

**9/12 复现，3/12 漂移（25%）**。两次调用相隔约 2 小时，`served_model` 在两批里都是 `DeepSeek-V4.1-Flash`，所以不是模型别名换了版本；这次审计无法把 25% 进一步归因到服务端路由还是采样。

### 3.3 只差一个小数位，4 次对照里 3 次翻转

两臂在同一个决策索引上常常共享同一段历史（anchor 相同、中间动作也相同），因此存在**内容几乎相同**的状态对。逐叶比对：

| 状态对 | 逐叶差异 |
|---|---|
| task1 d1（raw vs checked 的 state） | **1 处**：`history[0].training_seconds`，0.263115… vs 0.259974… |
| task2 d1 | **1 处**：`history[0].training_seconds`，0.532508… vs 0.559846… |
| task1 d2 | 7 处：`training_seconds` ×2、两组自己写的 `proposal.{reason,review,hypothesis}`、`evaluation_payload_sha256` |

在前两对（唯一差别是训练耗时的第四位小数）上分别问两种上下文：

| 状态对 | 上下文 | 答案 A | 答案 B | |
|---|---|---|---|---|
| task1 d1 | raw | `catboost_depth6_reg3.0` | `catboost_depth4_reg3.0` | 翻转 |
| task1 d1 | checked | `catboost_depth4_reg3.0` | `catboost_depth4_reg3.0` | 相同 |
| task2 d1 | raw | `word_char_C0.3` | `word_char_C0.1` | 翻转 |
| task2 d1 | checked | `word_char_C0.1` | `word_C0.1` | 翻转 |

**3/4 翻转。** 一个对候选优劣毫无信息量的墙钟字段，足以改变下一步动作。

## 4 这对原实验意味着什么

### 4.1 原实验的结构

三臂只共享 anchor（试验 1）。`task1` 顺序 local→raw→checked，`task2` 为 local→checked→raw。

| 任务 | 分叉发生在 | raw 的路径 | checked 的路径 |
|---|---|---|---|
| 表格 | 第 2 次模型决策（d2） | `reg3` → `depth6_reg10` → `depth8_reg10` | `reg3` → `depth4_reg10` → `depth6_reg10` |
| 文本 | 第 1 次模型决策（d1） | `C0.1` → `C0.05` → `stats_C0.1` | `C0.3` → `C0.1` → `stats_C0.3` |

（我上一轮曾判断"表格两组第一次决策就分岔"，**是错的**：表格 d1 两组都选 `catboost_depth4_reg3.0`，实测重放也确认。）

### 4.2 三个数字放在一起

| 量 | 数值 |
|---|---|
| 同 session 内重复同 prompt（被网关复用，不可用） | 24/24 一致 |
| 跨 2 小时、同字节 prompt | 9/12 复现（25% 漂移） |
| 只改训练耗时第 4 位小数 | 3/4 翻转 |
| raw vs checked，同一状态配对 | **7/12 不同** |

跨上下文的分歧率落在"重放一次就可能改口"的带宽里，而且 task1 d1 那条历史分歧在重放中**方向反了过来**。结论：**这批数据无法支持"检查与记忆改变了决策质量"**，只能支持"检查与记忆在这批轨迹上改变了决策，但改变幅度与模型自身抖动同量级"。

### 4.3 需要改的措辞

| 文件 | 位置 | 问题 |
|---|---|---|
| `docs/checked_loop_pilot_results_20261120.md` | 第 11 行表格、第 31 行、第 56 节 | "文本检查组比原始历史组高 1.6579 个百分点"应降级为"单次观测，落在复现性噪声内" |
| `docs/下一阶段执行计划.md` | 第 5 行 | 同一数字被当作本轮主要成果转述 |
| 同上 | 第 36 行 | 用 79、82 条外层预测差异解释收益，实际那些差异来自一次可能的抖动 |

另外，第 33 行"检查组相对原始组在表格上改对 37 条、同时新增 37 条错误"正好是净零换净零，本身已提示两组选的是同族不同深度模型，不是能力差异。

## 5 已经改掉的代码（未提交）

不改 `contracts.DecisionState`（会影响另一条 v2 线），只改本实验的投影：

```python
# superagent/checked_history_policy.py, decision_context()
for field in ('hypothesis_check', 'paired_error_comparison', 'explanation_audit'):
    row.pop(field, None)
# Wall clock is machine noise, not a property of the candidate. Two arms
# that ran the same trials differ only in the fourth decimal here, and a
# measured replay showed that this alone flipped the next action.
row.pop('training_seconds', None)
```

- 新增测试 `test_wall_clock_never_reaches_the_model`，断言两种上下文的历史里都没有 `training_seconds`。
- `tests.test_targeted_feedback / test_hypothesis_agent / test_architecture_contracts`：**39 项通过**。
- 离线复验：删掉该字段后，task1 d1 与 task2 d1 两对 state 在**两种上下文中都变成逐叶 0 差异**，也就是真正的同 prompt，后续再出现分歧就只能是 `checks`/`memory` 造成的。
- 注意 `replay_same_history.py --verify` 现在会对历史 12 条报 `history` 不一致，这正是被删除的字段，属预期；`--verify` 校验的是修复前的构造。

没有改 `check_expectation`、`targeted_feedback`、`contracts`，也没有动 main 分支。

## 6 我建议的下一步

| 选项 | 成本 | 能回答什么 | 我的看法 |
|---|---|---|---|
| A. 停止这条线，把审计写成报告结论 | 0 | 交付配置保留确定性的本地 4/3；说明 LLM 决策收益小于噪声 | **推荐**。与已有的本地对照、发布验证完全一致 |
| B. 修复后重跑 48 次，看 7/12 是否降到 3/12 附近 | ¥0.37，13 分钟 | payload 干净后的跨上下文分歧率 | 可选。但它仍然只是一次观测，救不回 +1.66pp |
| C. 正式设计：固定 payload、≥8 个外层划分、每格 ≥3 次独立抽样 | ¥5–15，数小时 | 一个能写进报告的分布级结论 | 只有在必须声称"检查有用"时才值得 |
| D. 补数值表格与普通单文本迁移 | 需数据，¥1–2 | 通用性 | 与本次审计独立，可另行推进 |

如果目标是"报告里的实验结论可信"，A 的收益最高：它把一次不可复现的观测转成一条量过噪声的负面结果，正好对应瓶颈 13。

## 7 想请你判断的四点

1. **噪声底线的口径**：跨时段 25% 漂移 + 微小扰动 75% 翻转，你认可把哪个当"实现门槛"？我倾向用"同一 payload 的重放分歧率"，因为它是可复算的。
2. **payload 还要不要继续收缩**：`classifier_fits`、`selected_iterations`、`error_slices` 里的浮点、`score_gap` 的末位，目前都是确定值，但任何会随机器变化的字段都可能在窄边界上起作用。是否统一约定"进入 LLM 的字段必须与运行环境无关"？
3. **网关复用响应这件事**要不要写进方法说明（影响以后任何"重复 N 次取多数"的设计）。我倾向于写，并在协议里要求"独立样本必须改变请求内容或显式禁用缓存"。
4. **是否接受 A**：把 1.6579pp 降级、保留三组轨迹作为机制演示（"模型会引用失败并改动作"这一条仍成立，且有原文引用），但不再声称准确率收益。

## 8 复现与文件

```powershell
# 零费用：证明重放就是历史 prompt
python runs\integration\scripts\replay_same_history.py `
  --run-dir D:\codex\python_withai\runs\checked_loop_pilot_v1 --verify

# 有费用：48 次调用，输出目录必须是新目录
python runs\integration\scripts\replay_same_history.py `
  --run-dir D:\codex\python_withai\runs\checked_loop_pilot_v1 `
  --run --output runs\replay_v3 --repeats 2
```

| 文件 | 说明 |
|---|---|
| `runs/replay_v2/replay.json` | 12 个状态 × 2 上下文的全部答案与稳定性标记 |
| `runs/replay_v2/llm_usage.jsonl` | 逐次 token、费用、`response_id`、`served_model` |
| `runs/replay_v2/events.jsonl` | 每次请求的完整 messages 与原始回答 |
| `runs/replay_v1/completion.json` | 索引 bug 造成的失败记录（保留） |
| `runs/checked_loop_pilot_v1/` | 被审计的原实验 |

费用明细：`replay_v2` 48 次调用、input 168 638 tokens、output 50 784 tokens、¥0.371774；`replay_v1` 2 次、¥0.010751。累计 ¥0.382525，均为课程折算价，学校账户实际扣费待核对。

**证据边界**：单 seed 43911、两个 released 任务的训练内划分、同一模型的两批请求（相隔 2 小时）、每个条件 2 个状态样本。不能外推到其他模型、其他时段或其他任务；本审计给出的是"这批数据不支持该结论"，不是"检查与记忆一定无用"。
