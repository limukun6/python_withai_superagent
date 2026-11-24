# B 给 A 的本轮交接

本地分支 `codex/b-symmetric-pilot` 从 A 的 `5ed06c66d972a50413201e3a2f2d9f74f404d214` 建立。新提交涵盖真正对称的文本方法、预算对照和 B 负责的四任务目录实验。生产入口、训练动作执行和费用账本继续由 A 维护。

## 能接什么

- 显式声明 `pair_semantics=symmetric`，提供两个文本列和两种分类标签。词级、字符级、词字符组合以及资源可用时的冻结 MiniLM 都执行交换不变的特征变换。两种标签可以是字符串。此类清单不声明哪边获胜。偏好契约仍按左右胜负处理，有方向的关系清单走保留列顺序的 `text_tabular`。本批真实对称语料的证据只覆盖二分类。
- `capabilities_for(spec, train_rows=...)` 继续返回可信的单次训练及重训上限。`next_action(state, capabilities=..., trusted_partitions=..., allocation='adaptive'|'fixed', on_explore=...)` 沿用六字段动作。默认动态。正式命令已支持本地动态，固定/动态实验用独立进程选择 B 的调度参数，随后用同一个 A 命令处理、执行守卫、选型与导出。A 可在自己的入口公开该参数，并在允许的 Flash 探索步骤传回调。
- 请重点审阅两个共同接点 `superagent/schema.py` 和 `superagent/task_semantics.py`。前者允许对称清单使用中性标签，后者按真实训练函数识别能否保持交换不变。方法目录、调度器和新方法的其余生产变更均在 B 模块。
- MiniLM 使用固定的 `all-MiniLM-L6-v2`，revision `524b52f1d626e9a5a16462e141fc9dffc25d5aeb`。将三份匹配摘要的文件放在本机，再设置 `SUPERAGENT_MINILM_WEIGHTS`。九次独立进程的搬迁、缺失和改动权重预测已验收。[权重交付实测](batches/BD_01_results.md)

## 已交付的实验

| 研究问题 | 实际观察 |
|---|---|
| 增加强方法是否有用 | 同一训练留出，Task1 准确率 73.45% → 74.86%，Task2 69.14% → 71.47%。短信确实尝试 MiniLM，最后仍选词字符方法。Dry Bean 同目录重现相同预测。[四任务配对表](batches/BQ_01_results.md) |
| 预算能否随反馈改变 | 16 条同预算上限的路径，固定共 88 次实训，动态 82 次。短信两个留出的 macro-F1 分别提高 2.70、2.37 个百分点。其余三任务对应逐行预测相同。[八次配对](batches/B1_08_results.md) |
| 下一步是否真的读了反馈 | 170 个实际动作逐字段重现。对四个任务的首次真实分歧，交换两份完整的训练内反馈都会双向改变动作。[对照记录](batches/B1_09_replay_results.md) |
| 对称句对能否迁移 | MRPC 两个开发留出开放 MiniLM 后均被自主选中，准确率增 1.21、3.16 个百分点；所有已导出的模型交换输入后逐行标签及每类概率均完全相同。[MRPC 实测](batches/B1_06_results.md) |

以上本轮 30 条研究路径，共 293 次实际监督训练，含最终重训。软件测试夹具另计。模型 API 请求 0 次，本轮新增费用 0 元。课程验证与正式测试未参加。目录和规则由开发者实现，本地控制器按任务契约和实际反馈在运行中选择。本轮没有付费 Flash 的效果结论。[目录图](../research/b_figures/catalogue.png) · [调度图](../research/b_figures/allocation.png)

## 联调所需的输入

1. 两套本机环境分别重现了 A/B 两套行号。共同输入要传源训练内容指纹、确切行号文件、候选和依赖，而不仅是种子。读取 B 的行号后，两环境的两任务、八个内外分区内容全部对齐。[跨环境检验](batches/BP_01_results.md)
2. B 的 C1 标题与 Dry Bean 两个任务块已经零训练预检。完整 12 条路径的顺序、候选、初始方法和行号已登记。当前共同入口的本地动态选项检查通过，raw/checked 加动态 AB 入口仍被拒绝。A 接好正式固定/动态选择、raw/checked 回调和同任务费用账本后，两人再共同冻结 C1 版本。[C1 预登记](batches/BC1_01_preparation.md) · [实际入口核对](../research/bc1_preparation/results.json)
3. H1 的 PAWS 和 H2 的 EuroSAT 已登记到确切源文件和行号。两者均为零训练、零评分。H2 有两个互不重叠的图像样本，均使用来自不同地理区域的训练与留出图像。等共同版本和唯一部署策略冻结后才运行 H 的正式比较。[H1 登记](../research/b1_05_pair_pilot/h1/completion.json) · [H2 登记](batches/BH_02_seal_results.md)

## 取包

在已有指定 A 基线的仓库中检查增量包，再取成独立的本地审阅分支：

```sh
git bundle verify AIAA3102-B-feedback-experiments-1121.bundle
git fetch AIAA3102-B-feedback-experiments-1121.bundle refs/heads/codex/b-symmetric-pilot:refs/heads/review/b-feedback-1121
git diff --stat 5ed06c66d972a50413201e3a2f2d9f74f404d214..review/b-feedback-1121
```

实验脚本、输入指纹、脱敏动作和紧凑回执随 Git 增量包交付。训练数据、冻结模型、权重、API 密钥及账本保留在各自的实验环境。
