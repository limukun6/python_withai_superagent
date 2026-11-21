# B 交给 A 的目录与调度器

适配器版本 **1.1**，动作协议仍为 **AB v1**。A 基线为 `67d9874ebb8fa2411ab142c94570078860c8cc34`，B 的调度策略沿用 `781e8c5efed8b263a3972643d7fa067fce6d77d1`。交接分支为 `codex/b-bridge-1121`。

## 在 A 的控制循环中接入

下面三行分别对应任务开始、新一轮状态准备和动作选择。`state["task"]` 是 A 的 `TaskSpec.record()`。训练行数取完整训练集，因为最终重训也必须能在同样的资源条件下运行。

```python
from superagent.ab_catalog import capabilities_for, untried_candidates
from superagent.ab_scheduler import next_action
from superagent.scheduler_bridge import dispatch_action

# 任务开始时注册一次，之后由执行器保管。
capabilities = capabilities_for(state["task"], train_rows=training_row_count)

# 每得到一条新观测就刷新，然后调用 B。
state["available_candidates"] = untried_candidates(capabilities, state["history"])
action = next_action(state, capabilities=capabilities, trusted_partitions=partitions)
outcome = dispatch_action(state, action, capabilities, allowed_splits, execute, logger)
```

- `capabilities` 保留完整的已注册配置。每项有 `kind`、`params`、`classifier_fits`、`refit_classifier_fits`、`supports_probabilities` 和 `swap_label_rule`。本版已入目录的配置均为每次评估一次监督训练、最终重训一次。类别字段交互树、图像梯度特征、冻结 MiniLM 加词法特征从第一步起按任务类型与本地依赖进入目录。MiniLM 需要校验通过的本地权重。核 SVM 还检查完整训练行数、内存和指标是否需要概率。当前目录只接收有明确左右偏好语义的二元文本对。注册集合为空时由入口记录不支持原因。
- 本地规划与可选 Flash 规划使用同一个 `untried_candidates` 视图。首个候选由两组共用的固定规则选出，后续依据当前试验结果和剩余额度决定探索、复核或停止。启用 Flash 时，A 的选择回调可通过 `on_explore` 接收本轮合规的候选 ID。API 使用量由 A 的同任务累计账本记录。
- `partitions` 是可信本地数据，按已登记的 split ID 保存 `row_ids`、`labels` 和 `evaluation_payload_sha256`，供本地核验。发往执行端的 `allowed_splits` 按相同 ID 关联划分描述与指纹。每次观测后，把执行器返回的实际预算、试验参数、完整 `metric_spec`、逐行预测与对应指纹放入下一轮状态。逐行证据同时支持 `candidate` 与 `candidate_id`。两者同时出现时必须一致。
- `explore` 从未试过的候选中选，`confirm` 从历史恢复已试候选的原参数，并使用新登记划分。B 返回的 `stop` 保留六字段格式，赢家字段为空。A 仍需检查执行状态与完整证据，再按共同选型规则重训和导出。B 提供 `selection_evidence(state, trusted_partitions=partitions)` 供本地选型使用。底层每次监督训练在调用前检查可用次数，执行结束后按实际次数对账。匿名脚本演示了这两步。

## 取包与复跑

在已有仓库使用下面的命令可把交接版本放入一个独立目录。先把 `AIAA3102-B-bridge-1121.bundle` 放在命令执行目录。最后一步使用项目的 Python 环境及一个新的输出目录。

```sh
git fetch origin codex/expanded-feedback
git bundle verify AIAA3102-B-bridge-1121.bundle
git fetch ./AIAA3102-B-bridge-1121.bundle refs/heads/codex/b-bridge-1121
git worktree add --detach ../b-bridge-1121-review FETCH_HEAD
cd ../b-bridge-1121-review
python -m scripts.run_b1_04_bridge runs/b1_04_local_01
```

## 这次验收到了哪一步

同一匿名任务中，A 的 `dispatch_action` 接收了 B 的 4 次探索、6 次复核和 1 次停止。前 10 次训练后还剩 1 次，A 用它完成了全训练集重训。两个全新的进程分别从原目录和迁移后的目录运行正式 `superagent predict`，31 行的行序、标签和输出文件摘要一致。新增 API 请求为 0。本轮只用了匿名合成输入。

第 9 个动作可以用来查验反馈是否参与了决策。逻辑回归即使赢下剩下的一次复核，也达不到两次胜出的晋级条件。B 把这次训练留给了新的 SVM 参数。它随后在两个共同划分上分别答对 35/51 和 36/51，同一划分上此前领先的树模型分别答对 30/51 和 33/51。B 采用了新配置。全库 389 项软件测试通过，含 7 项此次新增的接口与预算检查。

下一项双方验收由 A 从正式 `superagent run` 命令启动同一个匿名任务，让统一入口生成状态并调用这个适配器。对照动作分发日志、实训账本、冻结文件和新进程预测即可判断是否接通。随后再排 Task1、Task2 或其他任务的正式对照。

[预登记方案](batches/B1_04_plan.md) · [可复跑示例](../scripts/run_b1_04_bridge.py) · [动作与实训记录](../research/b1_bridge_1121/results.json) · [验收结果](../research/b1_bridge_1121/completion.json)
