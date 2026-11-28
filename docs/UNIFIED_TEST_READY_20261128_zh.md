# 同一入口的自主选模、机器适配与测试准备

这次合入 Task1 的类别编码器与融合研究，以及 Task2 的候选增补与融合研究。验收脚本按课程文件布局只取训练成员，分别封装成匿名输入。每次只给任务包、无标签待预测数据、同一份可用权重目录和相同预算。两个任务使用同一条命令启动，模型、融合比例和停止点都由程序在运行中决定。[Task1 来源](../research/task1_feedback_extension_1127/designated_release.json) · [Task2 来源](B_TASK2_HANDOFF_20261127_zh.md) · [事前登记](../research/unified_environment_freeze_1128/protocol.json)

## 程序怎样作决定

1. 读题目规定的指标和可用数据，识别字段类型、类别数和能否按组划分训练数据。
2. 探测这台机器可用的 CPU、内存、加速设备、软件和本地权重，再由数据形态和任务约束决定哪些实验有资格参与。普通表格、可交换左右顺序的二选一文本、图像有不同的可用操作。
3. 先取得一个可交付的基线。后续根据训练内部划分的逐行预测、模型之间的分歧、确认划分表现和剩余资源，选择继续探索、采纳组合、保留现有模型或停止。最终模型经全量训练和独立重载。开发者规定了通用操作和预算，两次运行只用训练内部反馈来选模型和权重，课程指定验证的核分发生在两个输出封存之后。[统一入口](../unified_agent.py) · [实验调度](../superagent/autonomous_run.py) · [训练与交付验收](../scripts/accept_unified_environment.py)

之前的实验也记录了两种相反的决策。Task1 的四组训练内对照都保留了原模型。另一个按电影分组的文本任务上，Agent 则选择 ELECTRA、GTE-base 和相似案例投票，独立留出从 348/450 升到 354/450。这支持程序能根据反馈改变选择。课程尚未公布的题型表现，需要以那次实际运行为准。[Task1 四组实验](batches/TASK1_CELL_ENCODERS_20261128_zh.md) · [电影分组实验](B_TASK2_PORTFOLIO_ADVISER_20261127_zh.md)

## 换一台机器会发生什么

队友此前在 RTX 5070 上遇到过驱动能看到显卡、PyTorch 构建却缺少对应计算内核的情况。新版在独立进程中对 CUDA 或 Apple MPS 做矩阵运算，并把结果读回 CPU。探测失败、驱动不兼容或显存不足时，可用候选和批量会相应调整。多个可用 CUDA 设备按剩余显存选一张。文本推理中途显存不足会先缩小批量，必要时在 CPU 上重算尚未完成的批次。表格、深度模型和基础数值库都受本机可用线程的约束，高配机器单项通常最多使用 8 个线程。未指定拟合额度时，程序根据 CPU 配额和可用内存选择 12、24 或 64 的上限。从第三次成功的试验起，调度器还按这台机器实测的拟合耗时更新计划。若下一次探索会挤掉全量重训时间，会优先重训已验证且来得及完成的候选。[资源探测与重试](../superagent/compute_resources.py) · [线程与显存故障用例](../tests/test_compute_resources.py) · [队友的 RTX 5070 记录](batches/TASK1_PROFILE_RESULTS_20261127_zh.md)

| 实际环境 | 任务与观察 |
| --- | --- |
| 本机 Apple M4，10 核、16 GiB | 检出 MPS，允许最多 8 个 CPU 线程。用 MiniLM-L6 对同一批 32 条文本在 MPS 和 CPU 各做一次推理，向量最大绝对差为 0.000000205。课程运行也记录到真实的 MPS 表格模型试验和文本编码。 |
| macOS arm64、Windows x64、Linux x86-64 的 CPU 环境 | 每个环境可用 2 个线程，无 PyTorch、Transformers、CatBoost 和预训练权重。相同的公开三分类任务给 133 行训练数据。三处均自动给出最多 12 次的额度，实际用 5 次，选出直方图梯度提升树。45 行留出均答对 42 行，balanced accuracy 为 94.07%，三个独立重载文件的 SHA-256 完全相同。 |
| CUDA 故障与低显存 | 受控测试覆盖 CUDA 声称可用却运行不了、设备可用显存不同、缩小批量和中途改用 CPU。Windows 和 Linux 上的本轮实机证据覆盖 CPU，NVIDIA 显卡的吞吐和实机故障恢复仍待有卡机器实测。 |

接收冻结预测包时，需要安装所选模型列出的兼容依赖。另一台机器若从头求解，可根据它实际拥有的工具选出不同模型。历史接收实验中，缺少 CatBoost 或文本权重时，备用模型曾有明显分数损失。因此跨机出结果与跨机保持精度需要各自核实。[Windows 和 Linux 执行记录](https://github.com/limukun6/python_withai_superagent/actions/runs/46119594472) · [旧版缺依赖实验](batches/B_RECEIVER_ENVIRONMENT_02_results.md)

## 为什么还要计算剩余时间

第一次整合验收中，Task2 在 30 分钟内观测到 25 次实际拟合，同一时间上限内交出了 940 行可重载的临时预测。但那份模型只用过 6,608 / 8,301 行训练数据。调度器给确认和重训留了拟合次数，却忽略了不同编码器在本机从冷启动运行可能相差数分钟。这份临时结果只用于定位问题。修正方案在查看任何 Task2 指定验证答案前登记，再以相同的 64 次、1,800 秒上限从头运行两个任务。原调度试过 22 条初筛路线才启动独立确认。修正后只完成前三条，便从实测耗时估算接下来约能安排 8 次拟合，提前锁定候选，开始在新的训练划分上比较。[时间保护规则](../superagent/compute_resources.py) · [事前恢复条件](../research/unified_environment_freeze_1128/protocol.json)

时间修正后的第一次全量重训也发现了一处两条代码线的接口差异。等权投票把权重保存为 `None`，表示两个成员各占一半。可选复核器曾把这个值当成列表，导致完整模型已写好却被报为部分完成。用当次已训练模型直接复现并修正后，又从头运行验收。后续复核也使用全量重训之后实际剩余的时间额度。最终的 Task2 运行中，调度器预估还能完成两次同等成本的拟合，随即停止探索，用全部 8,301 行重训两个成员。可选的相似案例试验至少还要四次拟合，程序保留等权组合。[投票权重解释](../superagent/analogy_review.py) · [回归测试](../tests/test_preference_analogy.py)

## 三次自主选择与最终交付

| 训练与资源条件 | 实际拟合 | Agent 选出的做法 | 课程指定验证 |
| --- | ---: | --- | ---: |
| Task1，同一通用入口，最多 64 次、30 分钟 | 36 | 三个种子的类别属性对齐 CatBoost 取平均 | 1,561/2,030（76.90%） |
| Task2，扩大资源目录，无预计算向量，最多 64 次、30 分钟 | 23 | GTE-small 与 GTE-base 各半 | 693/940（73.72%） |
| Task2，整合源码，恢复此前完整的通用资源目录及无标签文本向量，最多 22 次、30 分钟 | 20 | BGE-small 37.5%、GTE-base 37.5%、基于 GTE-base 的相似案例 25% | 710/940（75.53%） |

第三行从 8,301 条原始训练样本自主启动，启动参数未指定模型和融合权重。为与原实验相同，训练进程可以查找按编码器版本及文本内容指纹保存的无标签向量。另开进程关闭缓存，得到的 940 行结果与此前计分的三模型交付逐字节一致。第二行与第三行同时改变了资源目录、缓存和预算，17 道题的差距不能单独归因于其中一项。[首次普通整合验收](../research/unified_environment_freeze_1128/acceptance.json) · [整合源码按原条件复现](../research/unified_environment_freeze_1128/reproduced_710.json)

正式对照的 Task1 使用表中第一行的模型。Task2 使用第三行在当前整合源码下重新训练的三模型。它与此前交接的原始模型逐行给出完全相同的 940 个答案。另行验证旧模型也能在整合源码下直接装载。不过两次训练的部分数值系数有细微差异。为了把两组正式对照限定在最后使用的样本范围，两边均使用当前源码训练和预测。原始拟合文件另行保留。两份包内都有同一份最新预测源码。一个新任务进入普通入口时会根据它自己的训练反馈选择。历史交付文件、使用的硬件和可复用资源都在凭据中各自标明。[原模型交接](B_TASK2_HANDOFF_20261127_zh.md) · [跨版本独立重载](../research/unified_environment_freeze_1128/registered_task2.json)

## 最后重训与正式测试

| 任务 | 只用训练集 | 合并后重训 | 合并多用的行数 | 固定配方重训次数 |
| --- | ---: | ---: | ---: | ---: |
| Task1 | 8,117 | 10,147 | 2,030 | 3 |
| Task2 | 8,301 | 9,241 | 940 | 3（含重新建立相似案例索引） |

对照事前指定只用训练集为主结果。第二组沿用同一组模型、表征、参数、种子及投票比例，在原训练数据末尾加入相应验证集重新拟合。它已经看过指定验证答案，因此独立验证成绩留空。两组都在换目录的隔离进程中用无标签开发样本检查重载。冻结提交登记四个模型和两个交付包的校验值，随后才准备正式测试特征。每道题两组共用同一特征文件和行序。Task2 的两列标题与答案同处一个文件，特征步骤仅投影两列标题。四份预测封存后，独立评分步骤才取出两个任务的答案，计算两组准确率、改对及改错数量。正式逐行结果保存在本地。H2 继续封存。[事前方案](../research/unified_environment_freeze_1128/formal_test_protocol.json) · [两组模型冻结记录](../research/unified_environment_freeze_1128/frozen_release.json) · [评分入口](../tools/evaluate_frozen_released.py)

## 使用这版

面对一份新的训练任务，准备好环境后从仓库根目录运行。下面的 64 次拟合和 1,800 秒是两份课程训练包的通用无缓存验收预算。省略 `--fit-limit` 可以采用机器自己算出的额度。

```sh
python -m unified_agent solve training.zip --prediction-input features.zip --weights-dir weights --run-dir run --output predictions.csv --fit-limit 64 --time-limit 1800
```

按历史条件检验 Task2 时，先准备完整权重目录和该目录对应的冻结无标签向量缓存，再运行相同的入口。内部训练划分和方法沿用默认值。

```sh
SUPERAGENT_PAIRED_FEATURE_CACHE="$FEATURE_BANK" python -m unified_agent solve training.zip --prediction-input features.zip --weights-dir "$WEIGHTS_DIR" --run-dir run22 --output predictions.csv --fit-limit 22 --time-limit 1800
python -m unified_agent predict features.zip --weights-dir "$WEIGHTS_DIR" --run-dir run22 --output replay.csv
```

带成绩交付的 Task2 复现条件为原完整通用资源目录（见复现记录）、无标签编码器向量缓存、上限 22 次拟合与 1,800 秒。两种预算都会留下逐次决策日志。对已经选好并冻结的模型，解压任一交付包后进入对应的 Task1 或 Task2 目录，直接对无标签特征做预测。GitHub 分支保存完整训练入口，本地交付包带两个最终模型及其预测所需权重。若裸机尚未安装依赖，在同一 Python 3.11 环境先运行第一行。

```sh
python -m pip install -r requirements.txt
python -I predict.py --features features.zip --output predictions.csv
```
