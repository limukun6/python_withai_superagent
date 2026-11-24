# A151 迁移与复核验收结果：五条路径全部通过

2026-11-24。按 `docs/batches/A151_matrix_plan_20261124_zh.md` 执行。**5 条路径全部完成，17 次 fit、6 次 Flash、¥0.067218、没有新增 unknown。**

## 1 结果

| 路径 | 任务 | 调度 | fit | 调用 | 选中候选 | 验证 | 独立复核 |
|---|---|---|---:|---:|---|---:|---|
| w1_categorical | `released_task1`（类别表格） | ab-v1 | 3 | 1 | `catboost_depth6_reg3.0` | 0.7483 | — |
| w1_numeric | `dry_bean`（数值表格） | ab-v1 | 3 | 1 | `numeric_tabular_logistic_C0.1` | 0.9225 | — |
| w1_single_text | `sms_spam`（单文本） | ab-v1 | 3 | 1 | `single_word_char_C3.0` | 0.9474 | — |
| w1_single_text_grounded | `sms_spam` | ab-v1 | 3 | 2 | `single_word_C3.0` | 0.9347 | — |
| w2_confirm | `released_task2`（成对文本） | **ab-finalists** | **5** | 1 | `word_char_C0.05` | 0.7011 | **true** |

合计 **17 fit / 6 调用 / ¥0.067218**。每条路径都 `requested_planner_completed=true`、`attempted==completed`、无新 unknown、交付校验通过。

## 2 两条工作流各自证明了什么

**W1 迁移成立。** 压缩菜单表示在三种新任务类型上都能跑通（类别表格 15 候选、数值表格 5 候选、单文本 14 候选），grounded 契约也在单文本上成功：第 1 次提案因 `no_same_penalized_parameterization_and_observed_reference` 被拒，修复后 **7 条断言全部 confirmed** 并被接受。

**W2 独立复核成立。** `ab-finalists` 的动作序列是 `explore×2 → confirm×2 → stop`，`basis=["primary","confirmation_1"]`，两个 finalist（`word_char_C0.05`、`word_C0.05`）都在两个划分上完成测量：

| finalist | primary | confirmation_1 | 均值 |
|---|---:|---:|---:|
| `word_char_C0.05` | 0.6520 | 0.6573 | **0.6547** |
| `word_C0.05` | 0.6228 | 0.6507 | 0.6368 |

`independent_confirmation_observed=true`。这是 A151 首次真正走到独立确认分区训练。

## 3 过程偏差（必须记录）

| 次 | 事件 | 花费 | 处置 |
|---|---|---:|---|
| attempt_01 | 运行器漏传 `--api-key-file` / `--budget-ledger`，在发请求前报错 | 0 | 修正参数后重跑 |
| attempt_02 | 前两条路径成功；第三条卡在交付检查——`dry_bean`/`sms_spam` 是我自建的包，**没有 test split**，`superagent predict` 无法运行 | 2 调用 | 计划修正：无 test split 的包改用"新进程重载模型 + 验证集评分"，与运行记录逐值比较；补做 `w1_numeric` 的重载校验（2,723 行一致） |
| attempt_03 | 三条路径全部成功，但 `w2_confirm` 被我的验收逻辑误判失败——`independent_confirmation_observed` 在 `confirmation` 对象里，不在顶层 | 4 调用 | 修正字段读取；**离线重算**五条路径（`recheck.json`），不重跑 |

修正后的五条路径全部 `pass=true`（`research/a151_matrix/recheck.json`）。所有失败原件保留，未覆盖。

## 4 交付校验

| 路径 | 渠道 | 行数 | 一致 |
|---|---|---:|---|
| w1_categorical | `superagent predict`（CLI） | 2,537 | 是 |
| w1_numeric | 新进程重载 + 验证集评分 | 2,723 | 是 |
| w1_single_text | 同上 | 1,032 | 是 |
| w1_single_text_grounded | 同上 | 1,032 | 是 |
| w2_confirm | CLI | 3,232 | 是 |

## 5 能说什么、不能说什么

**能说**
1. 压缩菜单表示在三种新任务类型上可用，grounded 修复路径在单文本上可用。
2. 调度真正走到 `confirm` 时，独立确认分区的训练被观测到（2 个 finalist × 2 个划分）。
3. 五条路径的资源是 17 fit / 6 调用 / ¥0.067218。

**不能说**
1. **提分或迁移收益**：单任务单路径，没有对照分布；grounded 在单文本上（0.9347）反而低于 L0（0.9474），与之前成对文本同向，仍属单样本观察。
2. **通用性**：三条任务里两条是我自建的开发包，`released_task1/2` 也是已见任务。
3. **压缩省时**：仍然没有 full vs compact 的配对计时。

## 6 账本

| 项 | 之前 | 之后 |
|---|---|---|
| 请求数 | 400 | **406** |
| known 课程折算 | ¥7.195726 | **¥7.262944** |
| unknown / pending | 16 / 0 | **16 / 0（未增加）** |

## 7 复跑

```powershell
Set-Location -LiteralPath 'D:\codex\python_withai\runs\integration'
$py = 'D:\codex\python_withai\runs\a79_encoder_assets\venv\Scripts\python.exe'
& $py -m scripts.run_a151_matrix --output '<新目录>'        # 仅解析校验
& $py -m scripts.run_a151_matrix --ledger ... --authorization research/a130_accepted_history_v3/accepted_history.json `
  --api-key-file 'D:\codex\python_withai\api_key.txt' --only w2_confirm --output '<新目录>' --execute
```

证据：`research/a151_matrix/{plan,recheck}.json`、`attempt_02/`、`attempt_03/`，每路径含 `run/`、`predictions.csv` 或 `delivery_check.json`。
