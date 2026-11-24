# B 线接入与匿名验收

A 的接入基线是 `0b112d823a1991686281fdc88602e758e29a4202`，B 的工作分支为 `codex/budget-search`。首轮代码和证据在 `44568ac`，修正后的运行版本在 `26bb101`。调度器只使用任务结构、实际观测、合法候选和执行器给出的预算。A 保留正式入口、训练执行、预算持久化与最终导出。

如果收到的文件名为 `AIAA3102-B1-handoff.bundle`，把它放在已有 A 基线的仓库目录中，运行下面两条命令即可读取完整 B 分支。之后可以在自己的接入分支上审查差异。

```sh
git bundle verify AIAA3102-B1-handoff.bundle
git fetch ./AIAA3102-B1-handoff.bundle refs/heads/codex/budget-search:refs/heads/codex/budget-search
```

## A 需要接的代码

```python
from superagent.ab_catalog import common_catalog
from superagent.ab_scheduler import next_action, selection_evidence

menu = common_catalog(task_contract, train_rows=n_full_train)
state["available_candidates"] = menu
action = next_action(state, trusted_partitions=local_partitions)
```

- 两组在第一次决策前调用同一个目录。每项含 `id`、`params`、输入类型、表征、真实训练器声明、概率能力、交换标签规则和最大训练次数。A 的 `ExperimentProposal` 只接受 `id` 与 `params`，传给它的配置需仅投影这两个字段。确认可重复候选在新划分上的训练，单独分发。A 既有的禁止重复探索检查仍适用于 explore。
- 每份可信的本地划分含 `row_ids`、`labels` 和 `evaluation_payload_sha256`，有依赖关系时可另给等长的 `groups`。训练器为同一划分复用同一份不可变训练与评价输入。每次反馈都追加独立的 trial 编号和实际 `split_id`，让六字段动作能引用对应的历史。可用的确认划分以 `registered_confirmation_splits` 给出，支持空列表或依次为 `confirmation_1`、`confirmation_2`。
- explore 执行未尝试候选的 primary。confirm 执行指定候选和新划分。stop 交给 A 共同选型和重训。停止时调用 `selection_evidence(state, trusted_partitions=local_partitions)` 可取当前守住的模型、每次替换的逐划分差值和 trial 依据。验收脚本的 `finish` 将共同划分送入 A 的 `confirmation.decide` 和 `finalization.finalize`。未完成配对的单次高分不参与替换。
- Flash 组可用 `next_action(state, on_explore=choose_id, trusted_partitions=local_partitions)`。A 提供的 `choose_id(tuple_of_eligible_ids)` 只返回一个合法 ID。所需的已授权提示、目录元数据、已核验反馈从 A 的可见状态构建。本地训练行、标签和预测向量只给可信评价代码。共同首轮、confirm、stop 都不会触发回调。付费尝试由 A 在发送前登记到同任务累计实际账本，更新请求数后再构造下一状态。回调异常原样向上传递。每条路径最多 11 次实际分类器训练和 3 次 Flash，额度更小时服从传入的更小上限。计费沿用 A 已冻结的口径。
- 时间、钱和拟合的最终授权在 A 的执行端。B 会检查已耗墙钟、请求额度、实际尝试训练数和最终重训预留。历史消耗未知时先返回 stop 并要求核账。这种原因不能授权 A 在计数仍未知时另起重训。只剩重训额度时 stop 才是正常的最终重训交接。
- 固定组合、传统的原有搜索目录以及正式 `run --allocation` 仍由 A 管理。B 新增的 `ab-v1` 仅在共享目录和训练器解析层可用。正式 raw/checked 启用新调度需 A 接入上述分发，再用同一匿名包做双方正式入口验收。

## 选型规则与适用范围

先探索三个成功候选（资源不足可更早退出），以共同主划分上的最高分作为暂时领先者。以后的候选要在至少两个共同划分上得分更高，且所有共同划分的平均增益达到 0.002，才能取代前者。平分、只有一次高分或单份坏反馈都不足以换人。调度器会尝试现有强路线。实际可达、选中训练、跨任务有效是三个不同阶段。

accuracy 用共同评价行上预测的分歧初筛。macro-F1 在有本地标签时按同一组重采样索引分别重算两模型的 macro-F1，缺少本地标签仅按 0.002 的主指标门槛筛选。ROC-AUC 使用执行器记录的 ROC-AUC，硬预测相同也不能判定模型排名相同。新划分及行序与指纹绑定。重采样用于决定要不要复核，不能代替独立划分取得晋级资格。

类别表可选择 CatBoost 字段交互。图像可选择局部梯度方向直方图配逻辑回归。单文本与明确的二元偏好文本对可选择冻结 MiniLM 加词和字符特征。模型均支持新行预测和概率。数值表的树和核模型按数据规模及分数接口筛选。混合表和关系文本表保持 A 的类型适配。对称或语义不明的文本对目前没有通过 A 语义检查的相应 B 候选，返回空目录交回 A 处理。

MiniLM 只作下游特征提取。可选环境见 `requirements-ab-encoder-tested.txt`。将 `SUPERAGENT_MINILM_WEIGHTS` 指向含固定 revision 的 config、tokenizer、safetensors 的本地目录，训练和重载均需这些文件。目录先检查三份文件的字节数及 SHA-256。缺依赖或文件不匹配时隐藏该候选。验收在 CPU 上执行，未触发远端推理或下载。

## 验收与复跑

全库 372 项测试通过。另从留存状态重放修正版的 10 个决策，逐个比对输入摘要、六字段动作、本地分数与最终晋级依据，全部一致。源码指纹与验收版本 `26bb101` 一致。回归摘要见 `research/b1_ab_handoff/regression.json`。

```sh
python -m superagent.ab_scheduler examples/ab_v1/input.json
python -m unittest tests.test_ab_scheduler tests.test_ab_comparison
python -m scripts.run_b1_ab_acceptance runs/b1_new_directory
```

每个新目录只创建一次。B 的验收驱动通过 A 的训练数据入口、任务契约、分组划分、隔离训练器、共同选型与最终导出运行。实际的 `python -m superagent predict` 在两个新进程中分别从原目录和迁移后的目录运行。训练子进程额外直接计数分类器的 fit 调用，与逐次预算和训练器日志逐条核对。

| 同一匿名任务的两次开发记录 | 首轮 | 修正后 |
| --- | ---: | ---: |
| 探索 / 复核 / 最终重训 | 6 / 4 / 1 | 3 / 6 / 1 |
| 主路径实际训练 | 11 | 10 |
| 独立候选能力检查，另计 | 4 | 4 |
| 实际 Flash 请求 / API 费用 | 0 / ¥0 | 0 / ¥0 |

首轮最终采用一次主划分得分较高的候选。修正后，ExtraTrees 在 primary 比当时领先的 RBF SVM 低 3.92 个百分点，后两份确认划分分别高 1.96 和 7.69 个百分点，三个划分平均高 1.91 个百分点，晋级时附带 6 条 trial 记录。ExtraTrees 的三个训练内 accuracy 分别是 58.82%、64.71%、80.77%，等权平均为 68.10%。31 行无标签预测在两份新进程输出中与内存预测逐行一致，输出 CSV 字节也一致。

这一个匿名例子显示了替换规则如何被真实反馈驱动。修正版用了 6 次复核，只探索了 3 个候选。是否能用更少复核获得同样可靠的选择，留待双方正式入口通过后的 F/S 同预算对照。两个开发运行合计 21 次主路径训练，加 8 次独立探针。未访问课程验证集或正式测试集，本批没有新增付费消耗。可核查的 protocol、results、completion 分别保存在 `research/b1_ab_handoff/pilot/` 与 `research/b1_ab_handoff/accepted/`。
