# A151 多 seed 配对结果：grounded 在 6 个配对里输了 5 个

2026-11-24。按 `docs/batches/A151_multiseed_plan_20261124_zh.md` 预登记执行：2 任务 × 3 seed × 2 臂 = 12 条路径，**36 fit、15 次调用、¥0.238386**，无新增 unknown，25 分钟。

## 1 配对结果

### 成对文本（`released_task2`，accuracy）

| seed | plain | grounded | 差（pp） | plain 候选 | grounded 候选 |
|---:|---:|---:|---:|---|---|
| 707 | 0.6957 | 0.6755 | **−2.02** | `word_char_C0.1` | `word_C0.05` |
| 811 | 0.7011 | 0.6755 | **−2.55** | `word_char_C0.05` | `word_C0.05` |
| 823 | 0.6819 | 0.7011 | **+1.91** | `word_C0.1` | `word_char_C0.05` |
| **均值** | | | **−0.89** | | |

1 胜 0 平 2 负。

### 单文本（`sms_spam`，macro-F1）

| seed | plain | grounded | 差（pp） | plain 候选 | grounded 候选 |
|---:|---:|---:|---:|---|---|
| 707 | 0.9395 | 0.8684 | **−7.11** | `single_char_C3.0` | `single_word_char_C0.3` |
| 811 | 0.9474 | 0.9347 | **−1.27** | `single_word_char_C3.0` | `single_word_C3.0` |
| 823 | 0.9474 | 0.9347 | **−1.27** | `single_word_char_C3.0` | `single_word_C3.0` |
| **均值** | | | **−3.21** | | |

0 胜 0 平 **3 负**。

## 2 按预登记规则判定

| 判定项 | 结果 |
|---|---|
| 两任务均值 | pair −0.89pp、text −3.21pp，**都 ≤ 0** |
| 有 2/3 seed 为正的任务 | **没有** |
| **grounded_not_better** | **成立** |
| **positive_signal** | **不成立** |

规则是在看到结果前写的，不是事后拟合。

## 3 补充观察（不改变判定）

1. **两臂从未选中同一个候选**（6/6 配对候选都不同），所以差异全部来自选型，不是训练噪声。
2. **grounded 用的调用更多**：9 次 vs 6 次，因为部分路径触发了语义修复。
3. **不是"必然更差"**：pair 的 seed 823 grounded 反而高 1.91pp。方向不一致说明这是选型抖动，而不是系统性变好或变坏。
4. 单文本 seed 707 的 −7.11pp 是最大单点损失：plain 选了 `single_char_C3.0`（0.9395），grounded 选了 `single_word_char_C0.3`（0.8684）。

## 4 能说什么、不能说什么

**能说**
1. 在这 6 个同预算配对里，grounded 的选型有 5 次更差，两任务均值为负。
2. 预登记判定"grounded 不优于普通"成立。
3. 资源是 36 fit / 15 调用 / ¥0.238386。

**不能说**
1. **不能说是普遍规律**：只有 2 个任务、3 个 seed，且三个 seed 共用同一份验证集，不是三个独立任务。
2. **不能说是因果机制**：grounded 与 plain 的差别包含"是否强制断言"和"是否触发修复"两件事，本轮没有拆开。
3. **不能说普通路径更好**：本轮没有测普通路径与固定候选的对照，只测了两条路径彼此。
4. **不做显著性声明**：n=3，没有区间估计。

## 5 对定位结论的影响

之前基于两条单样本路径给出的"保留 grounded 为可选、不晋级默认"的建议，现在有 6 个配对支持，方向一致（5/6 更差、两任务均值为负）。建议不变，但依据从"两条单样本"升级为"一个预登记的 6 配对比较"。

## 6 账本

| 项 | 之前 | 之后 |
|---|---|---|
| 请求数 | 406 | **421** |
| known 课程折算 | ¥7.262944 | **¥7.50133** |
| unknown / pending | 16 / 0 | **16 / 0（未增加）** |
| 账本 SHA-256 | `cbfec185…` | **`bafe5608…`** |

本批新增 15 次请求、¥0.238386。

## 7 复跑

```powershell
Set-Location -LiteralPath 'D:\codex\python_withai\runs\integration'
$py = 'D:\codex\python_withai\runs\a79_encoder_assets\venv\Scripts\python.exe'
& $py -m scripts.run_a151_matrix --plan research\a151_multiseed\plan.json --output '<新目录>'   # 仅校验
```

证据：`research/a151_multiseed/{plan.json,analysis.json}`、`attempt_01/<path>/run/`、各路径 `delivery_check.json` 或 `predictions.csv`。
