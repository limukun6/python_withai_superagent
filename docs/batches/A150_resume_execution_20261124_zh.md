# A150 续跑执行：已接通，零成本验证通过

2026-11-24。上一轮交付了只读判定（`resume-status`），本轮接通**真正的续跑执行**：`--resume-from`。

## 1 做了什么

| 变更 | 内容 |
|---|---|
| `superagent/resume.py` | 新增 `adopt(parent, target, ...)`：把一个"可续跑"的中断件搬到新目录，抄 `trial_*.joblib`、`history.json`、`classifier_fits.json`、`prediction_evidence.json`、划分负载，并写 `resume_from.json` 记录来源与父清单 SHA |
| `superagent/ab_run.py` | 新增 `--resume-from`：先跑一遍 `inspect`，不通过就拒绝；把抄来的 trial 作为起点，预算计数从"已用"开始，执行器的 fit 记录也用父运行的真实记录预置 |
| `superagent/cli.py` | `run --resume-from <parent_run_dir>` |

### 五条硬约束（都已落地）

1. **必须新目录**：`adopt` 拒绝把父目录当目标，原件保持只读。
2. **必须同 fit 上限、同 seed**：父清单不同即拒绝。
3. **必须源码一致**：`inspect` 比对 `source_files`，代码变过就拒绝。
4. **必须任务包一致**：比对 `task_archive_sha256`。
5. **必须预算未结清**：账本有 unknown/pending 就拒绝。

## 2 零成本端到端验证（0 API）

用本地决策（`--planner local`，不调用模型）在 `sms_spam` 上做三步：

| 步骤 | 结果 |
|---|---|
| ① 一次跑完（基准） | 选中 `recipe_single_text_latent_word_logistic_…`，验证 **0.920173080213678**，3 fit |
| ② 人为中断件 | 只保留 trial 1 的证据（1 fit），删掉后续 25 个产物 |
| ③ 续跑 | 选中**同一候选**，验证 **0.920173080213678**，fit 计账 **3**（1 继承 + 2 新增），状态 `completed` |

**关键性质成立**：中断时 1 fit + 续跑 2 fit = **3 fit**，与一次跑完完全相同；结果逐位相同。没有重复 fit，也没有重复请求（本次本来 0 请求）。

产物：`research/a150_resume/{base_local,interrupted_local,resumed_local}/`。续跑目录里 `trial_1.joblib` 是继承来的、`trial_2.joblib` 是新增的，`resume_from.json` 记录父目录与清单 SHA；父目录始终只有 `trial_1.joblib`。

## 3 两个被护栏拦下的真实案例

1. **源码漂移**：第一次尝试续跑时我刚改过 `resume.py`，护栏直接拒绝——`executor sources changed since the run: ['resume.py']`。这正是它该做的事。
2. **执行器账本不一致**：第 2 次尝试报 `Dispatcher ledger differs from actual worker fits`——预算说已用 1 次 fit，执行器自己的派发计数却是 0。修法是把父运行真实的 fit 记录预置进执行器，而不是伪造占位记录（伪造会让 `classifier_fits.json` 出现假 worker）。

## 4 尚未做

1. **没有真实 API 场景的续跑验证**：本轮用本地决策验证，未验证"续跑后 LLM 决策与一次跑完一致"——因为还要额外花请求，且模型本身有漂移，逐位一致本来就不该期待。
2. **幂等键仍靠账本预留**：续跑不会重发已结算请求，但"同一请求只结算一次"的显式去重键仍未实现。
3. **`--planner checked-history` 的续跑未接线测试**：代码路径相同，但没有实际跑过带 LLM 的续跑。

## 5 测试

`tests/test_resume.py` 共 14 项，新增 3 项针对 `adopt`：抄写并记来源、拒绝已完成的运行、拒绝覆盖父目录。全仓 142 项测试通过。

## 6 资源

0 fit / 0 API / ¥0（验证用的都是本地决策）。账本仍 453 请求、known ¥7.882065、unknown 16、pending 0。
