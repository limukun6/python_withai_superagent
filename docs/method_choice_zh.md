# 分类方法与训练设备的自主选择

项目允许 SuperAgent 使用 Qwen 或 DeepSeek 作语言模型。按本组澄清的口径，下游分类器可以使用其他通用预训练模型提取特征。这里把 MiniLM 作为冻结的特征提取器，记录前向次数、输入 token、设备和秒数。本地运行没有服务商扣费。Qwen 本地向量仍沿用已有账本记录一份按公开单价折算的数值，便于和上一轮比较。规划器使用本地决策规则，本实验的生成式语言模型调用为零。

## 从统一入口运行

安装项目的 `requirements-pretrained.txt` 后，在仓库根目录执行以下命令。指定新的输出目录。

```sh
python -m superagent run "/path/to/task.zip" --run-dir "/path/to/run" \
  --planner local --method-choice auto --trials 7 --portfolio expanded \
  --time-limit 1600 --trial-timeout 480 --feature-cache run
```

默认允许在选中后下载公开的 MiniLM 权重，缓存位于输出目录父级的 `.superagent-weights/all-MiniLM-L6-v2`。也可以通过 `--pretrained-config` 提供 JSON，把 `minilm_weights` 指向已有目录，或设为 `null` 关闭。仅用 MiniLM 无需语言模型密钥或项目账本。选配 Qwen 类别向量时，提供 `tabular_weights`、已有项目的 `ledger`、`scope` 和 `cap_rmb`。使用 `SUPERAGENT_OFFLINE=1` 时，缺少本地权重的编码器跳过。常规训练入口可以在选择冻结后评分任务包提供的验证集。本轮对照驱动另有只读训练成员的约束。

## 哪些决定由 agent 作出

- 最多先运行五次原有自适应搜索。根据输入模态、可用样本、训练和内部搜索结果，在两个后续名额中选择继续原搜索、尝试从零训练的 MLP、类别共现向量、MiniLM 或 Qwen。表格编码器保留同一领先分类器和原始类别列，只增加由训练集类别名称得到的数值特征。少于五个普通候选时先完整运行可用候选，可行方法穷尽后提前结束。
- 原路线的领先者保证进入最多三名的复核。最终按三个训练内划分的平均任务主指标选模，并在当前可用的训练行上重训。方法选择的输入没有任务名或数据集名。
- `fixed` 用固定优先次序填充可行方法。`adaptive` 看共同错误、非线性模型与线性模型的分数差，以及上一轮挑战者相对于原领先者的表现。`paired` 在准确率任务上还计算同一批样本的改正和新增错误数。相差一个百分点且改变 40 行与相差一个百分点且改变 210 行，可以触发不同的下一步。该规则是预先固定的探索策略。
- MLP 随机初始化，仅用当前训练标签拟合。每次拟合先在 CPU 和可用的 CUDA 或 Apple MPS 设备上，按当前批次和网络形状测量前向、反向和参数更新。只有预计加速至少 10% 才采用加速器。训练后保存 NumPy 权重，可在 CPU 预测。其他预训练编码器的本地前向和神经网络的梯度训练分别记账。

MiniLM 使用公开的 `sentence-transformers/all-MiniLM-L6-v2`，冻结 BERT 权重、掩码平均池化、L2 归一化，产生 384 维向量。类别表在每个划分的训练类别上拟合最多三个主成分，预测时可直接查表。单段文本使用句子向量，明确声明为偏好关系的文本对使用左右向量之差，保留词、字符和适用的样式特征。可读类别名和英文兼容文本的判断均只使用输入本身。[模型卡及许可证](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

`auto` 根据可用表示选择表格或文本决策方式。另有可复查的深度对照。运行参数 `--method-choice encoder-depth` 让训练反馈决定是否试用 MiniLM-L12。在 `--pretrained-config` 的 JSON 中提供 `deep_minilm_weights` 的本机目录，例如 `{"deep_minilm_weights":".superagent-weights/all-MiniLM-L12-v2"}`。联网时，选中后才获取并核对缺失文件。使用离线模式时只考虑本地已有的权重。搬移文本模型后，预测的 `--minilm-weights` 应指向该模型实际选用的 L6 或 L12 目录。逐折成绩、下载量和额外试验的取舍见[主线实验记录](mainline_iterations_zh.md)。

可选的 `--method-choice encoder-mixture` 使用同一套训练反馈分配试验名额。只有任务指标为二分类准确率、L6 和 L12 都进入三名复核，且其中一个胜出时，Agent 才考虑合用两者。它从三个内部划分保存的概率学习一个权重，最小化预测真实标签的对数损失。出现在多个内部划分的同一行，合计只计一次。权重确定后，Agent 用该折全部训练行重训胜者，并保留另一个编码器在首次搜索时拟合的分类器。每折依然有 14 次分类器拟合。使用两者时，首次预测需要分别计算两个模型的向量。训练目录的 `encoder_mixture.json` 给出决策所依据的行数、权重和入围方法。

搬移组合模型时，复制 `selected_model.joblib`、`task_spec.json` 和 `final_selection.json`，然后将两个公开权重目录传给预测入口。

```sh
python -m superagent predict "/path/to/new-input.zip" --run-dir "/path/to/moved-model" \
  --minilm-weights "/path/to/all-MiniLM-L6-v2" \
  --deep-minilm-weights "/path/to/all-MiniLM-L12-v2" --output "/path/to/predictions.csv"
```

## 复核与分工

`method_decision_*.json` 保存每个额外试验开始前的可行方法、实际观测和选择原因。`trials.json`、`confirmation.json` 和 `final_selection.json` 可以连接实际训练配置、入围理由和重训产物。`downstream_usage.jsonl` 记录 MiniLM 的前向计算，`model_loads.jsonl` 记录下载和校验，`llm_usage.jsonl` 保留原有的 Qwen 账本凭据。

首轮五折冻结在提交 `cdbed4d`。用 `python -m tools.method_device_study --help` 查看实验入口，用 `python -m tools.audit_method_device --help` 查看逐行重放、屏蔽部分反馈和设备复测的入口。第二轮前三折对应 `efe9c34`，追加的两折对应 `7f85839`，两批运行的 agent 源码完全一致。各轮都可用 `--protocol-only` 先冻结协议。每阶段的对照方式、复现参数见[实验结果](task1_method_device_results_zh.md)。

运行 `python -m tools.method_smoke --output /path/to/new-smoke` 可在匿名任务上检查统一入口、模型重载和预测行序。加 `--minilm-weights /path/to/all-MiniLM-L6-v2` 后，会再用匿名类别表走一遍完整的本地 MiniLM 训练和预测，仅需已下载权重。

本轮按架构与运行分工实现候选、实验驱动、预算和日志，供测评与交付成员独立复核。Codex 参与实现、运行、核对和文档整理，开发者定义允许的能力和实验方案。每条实验轨迹由冻结程序独立作出决定。联网下载发生在选择之后，运行中的能力发现限于已实现的模型与训练方法。[团队计划](ten_day_experiment_plan_zh.md)
