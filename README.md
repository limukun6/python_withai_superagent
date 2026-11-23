# AIAA 3102 · General Algorithm SuperAgent

## 队友整合用的六组对照与独立标注

Task2 和 Dry Bean 共 12 份预测、八类 48 个检查案例，以及 24 份带逐条依据的遮蔽标注已归档。交接页列出两组提前结束的原因和完整组的成绩，并附标准库复核命令。[B 线补交与数据入口](docs/batches/B_C1_D3_handoff_20261123_zh.md)

## 11 月 23 日已验证的 Task2 方法

标准 Agent 在有十二层 MiniLM 可选时，自主选择了十二层 MiniLM 加词法与风格特征。它在训练集内按共享标题分组预留的 1,693 对上答对 **1,224 对（72.30%）**。另一组多提供了能学习标题改写片段的候选，Agent 实际训练并复核了它，最后也保留普通十二层。两组各拟合 10 次，模型 API 请求 0 次。四份独立进程导出逐行一致。[本分支可运行源码包](releases/verified-preference-2026-11-23/) · [实验成绩与复现命令](releases/verified-preference-2026-11-23/docs/batches/B_verified_preference_1123_zh.md) · [供队友整合的固定源码提交](https://github.com/limukun6/python_withai_superagent/tree/c6584776be99a1eaee59b881047566219651edbd)

## 11 月 22 日 A/B 整合

最新 B 线代码供队友从 `codex/b-a31-joint` 整合，当前交接提交为 `fa58e85`。候选目录、调度、验收记录及各实验分支入口见交接说明。下文记录 v0.42 路线在课程指定验证集上的历史成绩。[最新版 B 线交接](https://github.com/limukun6/python_withai_superagent/blob/codex/b-a31-joint/docs/B_HANDOFF_20261122.md)

## 11 月 20 日课程指定验证集

本次直接加载上一轮冻结的三个模型，先保存全部预测，再核对课程验证标签。

| 任务 | 冻结方案 | 正确数 | 验证准确率 |
|---|---|---:|---:|
| Task1 · 车内优惠券 | CatBoost，树深 6 | 1,511 / 2,030 | 74.43% |
| Task2 · 标题选择 | 轻量默认，MiniLM-L6 加词和字符特征 | 682 / 940 | 72.55% |
| Task2 · 标题选择 | 可选新方案，MiniLM-L12 与 L6 概率加权 | 682 / 940 | 72.55% |

双编码器把轻量模型的 39 个错误改对，也把另外 39 个正确答案改错。两种方案本次命中数相同，双编码器处理的本地特征 token 数是轻量方案的两倍。两项任务的模型 API 调用与费用均为零。正式测试集保留到最终评估。[逐行结果、版本及复跑说明](validation/mainline-2026-11-20/)

## 七次搜索的统一实验入口

使用 `--planner local --method-choice auto --trials 7 --portfolio expanded` 可以从相同的五次常规搜索开始，让 agent 根据数据和逐行试验反馈分配最后两个名额。符合条件的文本可以使用已冻结的 MiniLM 句子特征，普通分类路线始终参加搜索。两项课程任务在训练集内部的同预算对照、候选上限、费用及可重载模型验证见[七次搜索主线实验记录](docs/mainline_iterations_zh.md)。下文保留整合分支的四次搜索和先前发布成绩。

完整训练、权重目录配置和预测命令见[分类方法与训练设备](docs/method_choice_zh.md)。将选中的 MiniLM 文本分类器搬到其他机器后，可在预测命令中提供 `--minilm-weights /path/to/all-MiniLM-L6-v2`。预测输出旁的记录包含本次实际使用的特征 token 数。

这版在发布训练集内部按组跑五折。三套划分中，表格主线各答对 4,801、4,779、4,788 行（每套 6,494 行），文本小模型主线各答对 4,623、4,644、4,609 行（每套 6,608 行）。同一批行换组会改变难度。真实标题向量与保持其他训练步骤不变的纯文字分类器另有逐折对照。[七次搜索的逐折结果](docs/mainline_iterations_zh.md)

`--method-choice encoder-mixture` 允许 Agent 根据训练反馈尝试更深的编码器，并从三个内部划分学习怎样结合两个模型的概率。`encoder-mixture-wide` 让表现接近原领先者的小模型也能触发下一次探索。两种选项和默认的 `auto` 都可独立训练、导出和预测。[运行与权重配置](docs/method_choice_zh.md)

本包还附上同一版本 `auto` 用全部开发行重训后导出的模型。Task1 选中深度 6 的 CatBoost，Task2 选中 L6 句子向量加词和字符特征。两个目录均可直接作为 `predict --run-dir` 的输入。Task2 另传 `--minilm-weights /path/to/all-MiniLM-L6-v2`。每个模型都用 12 条新造的无标签输入核对过搬移前后的输出。正式课程评估依照发布安排另行执行。[Task1 模型](solutions/task1/development/) · [Task2 模型](solutions/task2/development/) · [训练与搬移记录](research/development_pool_models.json)

Task2 也附上宽松探索策略从全部开发行独立生成的模型。它在 3,256 条不同的内部验证样本上，为重训的 L12 学到 78.22% 的权重，其余 21.78% 给首次搜索保留的 L6。预测时提供 `--minilm-weights` 和 `--deep-minilm-weights` 两个目录。[双编码器模型](solutions/task2/development-mixture/)

宽松策略独立完成三套训练内五折，分别答对 4,661、4,689、4,659 对标题，准确率为 70.54%、70.96%、70.51%。相比小模型分别多答对 38、45、50 对，训练和首次外层预测由本地编码器处理的输入 token 数均为两倍。三套划分重复使用同一批 6,608 行。原先规定的晋级条件还要求超过更强的深模型对照，因此保留轻量默认，把宽松组合作为单独可运行的选择。[宽松路线的逐折核验](research/final_wide_encoder_results.json) · [默认路线的同版复核](research/release_unified_policy_results.json)

## 整合分支的实验与复现入口

当前统一配置为本地 `interleaved_routes`、四次搜索、前三名三划分复核、全训练集重训。两个发布任务使用相同配置。整合分支的发布验证、本地复验和阶段性实验协议见下列文档。此节之后的 v0.42 结果对应仓库随附的历史模型和报告。

- [预算复验和当前发布验证](docs/search_budget_replication_results_zh.md)
- [搜索分数反馈与实际复核耗时的配对协议](docs/current_stage_10h_zh.md)
- [搜索反馈与复核开销的实测结果](docs/feedback_value_results_zh.md)
- [历史决策证据复核](docs/decision_evidence_audit_zh.md)

分类方法的研究分支可用 `--method-choice paired --trials 7`。本地规划器按各次试验改对和改错的样本选择下一步，可用从零训练的神经网络，以及冻结的 MiniLM 或 Qwen 特征。神经网络训练前会实测可用设备。[使用方法与分工](docs/method_choice_zh.md) · [Task1 同预算对照](docs/task1_method_device_results_zh.md)

从仓库根目录使用同一个配置处理任务包。训练和正式预测是独立命令。按课程安排取得正式预测输入后执行第二条。两个命令分别需要新的运行目录和预测文件路径。

```powershell
python -m superagent run 'TASK.zip' --run-dir 'runs/reproduction' --planner local --policy interleaved_routes --portfolio expanded --trials 4 --seed 707 --trial-timeout 90 --time-limit 600 --feature-cache run
python -m superagent predict 'TASK.zip' --run-dir 'runs/reproduction' --output 'predictions.csv'
```

配对实验和发布验证使用 Python 3.14.4、NumPy 2.5.4、SciPy 1.18.1、scikit-learn 1.9.1、joblib 1.6.0、CatBoost 1.2.10、Requests 2.34.2 和 Pillow 13.0.0。运行前核对输出目录的 `protocol.json` 与本机依赖。按以下顺序执行两阶段对照，复验自动读取开发阶段的事前门槛判定。

```powershell
python -m tools.feedback_value_study --task1 'TASK1.zip' --task2 'TASK2.zip' --output 'runs/feedback_value' --stage development
python -m tools.feedback_value_study --task1 'TASK1.zip' --task2 'TASK2.zip' --output 'runs/feedback_value' --stage replication
```

## v0.42 历史交付快照

v0.42 · 课程开发验证快照 · 两个任务共用代码、预算和策略，无按任务名称选择解法。主交付是在验证前指定的 **fixed**：根据训练内反馈探索四个候选，在三个训练内划分复核前三名，再用完整训练集重训。**adaptive** 是比较组，按已有证据探索、复核或停止。两者均为本地规则规划，本次不需要 API key。

| 任务 | 验证行数 | fixed | adaptive | 拟合次数（含重训） |
|---|---:|---:|---:|---:|
| Task1 · 车内优惠券推荐 | 2,030 | 1,516 正确 · 74.68% | 1,516 正确 · 74.68% | 11 / 10 |
| Task2 · 成对标题偏好 | 940 | 656 正确 · 69.79% | 656 正确 · 69.79% | 11 / 10 |

两个任务都是逐行预测相同。本次验证集历史上已用于开发；这些成绩不是盲测或正式测试。已在训练内观察到自适应规则会过早排除有竞争力的候选，因此保留固定策略作为主交付。

## 从这里开始

- [本轮报告（PDF）](report.pdf) · [精确计数和费用](validation/scores.json) · [验证前冻结的协议与源码哈希](validation/protocol.json)
- **可直接预测的模型**：`solutions/task1/fixed/` 和 `solutions/task2/fixed/`，各有模型（含预处理）、模型说明及逐行验证预测。相邻的 `adaptive/` 保存比较组，而非按任务择优拼装的提交。
- `validation/task*/` 是本轮四次训练的决策、划分、冻结记录和访问日志。训练进程没有验证数据，其 summary 验证字段为空是预期行为；最终评分在上述独立 scores.json。公开日志中的本机路径已替换为相对路径，未改分数或决策。
- [三组训练内对照](research/inner/REPORT.txt) · [内部复核结果](research/inner/verification.json) · [AI 使用说明](AI_USAGE.md) · [代理功能文档](AGENT_GUIDE.md)

## 环境与复跑

在项目根目录执行。实测 Python 3.14，基础依赖的确切版本见 requirements-tested.txt；普通安装也可使用 requirements.txt。大模型服务和神经网络不是重放本次结果的前提。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m superagent --help
```

将课程原始 ZIP 放在你有权使用的本地目录，保持原文件名。在下列命令中替换 `/path/to/course-zips`。使用一个不存在的输出目录，避免覆盖结果：

```bash
python tools/released_validation.py freeze --packages "/path/to/course-zips" --out runs/reproduction
python tools/released_validation.py task1 --packages "/path/to/course-zips" --out runs/reproduction
python tools/released_validation.py task2 --packages "/path/to/course-zips" --out runs/reproduction
python tools/released_validation.py score --packages "/path/to/course-zips" --out runs/reproduction
```

四个模型全部冻结后才打开课程验证成员；不读取正式测试输入或标签。在相同环境比较输出行和正确数，不要求新训练的文件哈希相同。

**直接复核随包模型**（不训练）：`python tools/check_saved.py --packages "/path/to/course-zips"`。它用课程验证成员复算四份结果，并把仅含验证特征的临时包交给标准预测入口，验证导出的逐行预测。临时包的内部成员名遵循预测接口约定，与课程正式测试集无关。将来获准对新输入做正式预测时，入口为 `python -m superagent predict INPUT.zip --run-dir solutions/task1/fixed --output predictions.csv`，替换为相应任务目录。

**回归测试**：安装 `requirements-neural-tested.txt` 后运行 `python -m unittest discover -s tests`。本轮冻结代码的完整回归为 280 项通过。使用文件包时请先核对 SHA256SUMS；只加载可信的模型文件。本轮报告与日志只覆盖已声明实验，不包含历史项目的 API 总账、原始课程数据、密钥、缓存或正式测试预测。

After observing shared errors, the local policy can try frozen Qwen representations of text or readable category names. The original leader remains eligible for confirmation. [Setup and decision rules](docs/pretrained_choice.md) · [Task1 five-fold results](docs/task1_category_encoder_results_zh.md)
