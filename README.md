# AIAA 3102 · General Algorithm SuperAgent

## 11 月 27 日研发版

本地整合版在两个课程原包上分别复现 **75.52%** 和 **72.77%**。三个独立公开分类任务均完成同预算反馈对照和新进程重载。图像任务发现了可复现的难点，测得的两种干预均未通过采用门槛。[整合版结果、资源与复现方式](docs/JOINT_FEEDBACK_RESULTS_20261127_zh.md)

统一入口现在能在首次拟合前交付逐行预测，可选的反馈模式会从真实训练错误提出下一项核查。两份课程原包和三种公开分类任务已在本机复跑。[运行与验收结果](docs/RELIABLE_DELIVERY_FEEDBACK_1127_zh.md)

**11月27日，A线主线纠正为失败后自主恢复。** 用户明确要求LLM查证、修复或换路线使任务继续，不能以“阻断正确”替代完整Agent交付。已审计跨阶段恢复的缺口，并解除内部AUC的一条误阻断；完整RecoveryController仍待实现。本批0fit/0API。[恢复架构与主验收](docs/batches/SA18_RECOVERY_REALIGNMENT_20261127_zh.md) · [首项纠偏证据](docs/batches/SA18_RECOVERY_CORRECTION_RESULTS_20261127_zh.md)

**11月27日，字段疑问由实际任务判定是否阻断。** LLM报告问题字段，执行器核对类别数、指标和类型，保留不适用的疑问继续执行。292项检查，真实受控面板1放行/2阻断均符合预期，0fit/3API/课程¥0.017223；尚未证明自然任务节省费用。[结构、原始误阻断与证据边界](docs/batches/SA17_scoped_questions_results_20261127_zh.md)

**11月27日，人用CLI默认自动读题。** 本地能读输入也会在模型已配置且有文档时复核输出要求；显式manifest输入保持权威。274项检查，裸solve自主完成原ID、三类映射、指定类别概率和管道/BOM输出，2fit/2API/课程¥0.008978、27.781秒。程序化API和显式fallback保留旧默认，不代表独立未知任务全成功。[结构、兼容约定和实际证据](docs/batches/SA16_auto_semantic_review_results_20261127_zh.md)

**11月27日，无manifest数组与图像任务接入通过。** LLM从原文选择非标准NPZ成员/键、图像目录、指标和像素处理；正式proactive solve两条首次提案均通过。245项检查，新增4fit/2API/课程¥0.015720，54行新进程预测逐字节一致。结果是开发输入覆盖改善，未比较准确率。[结构与证据边界](docs/batches/SA14_native_semantics_results_20261127_zh.md)

**11月27日，A线正式实时读题和工具闭环验收通过。** 229项检查；Flash自主请求预测字段，再填写正确ID输出契约，正式solve完成2次训练与60行独立进程预测。2次API，课程¥0.008981，52.375秒。此前SA12失败保留；这是单条已使用开发任务的部署证据，非任意未知任务保证。[结构、实际工具轨迹和资源边界](docs/batches/SA13_tool_source_results_20261127_zh.md)

## 11月26日A/B共同研发集成

SA11新增正式`--semantic-artifact-from`任务解释复用：187项相关检查及两条独立solve通过，新增4fit/0API。它复用已验证读题结果，仍会新做预算内训练，不等于恢复旧搜索或保证未知任务全成功。[正式用法与验收边界](docs/batches/SA11_semantic_artifact_results_20261126_zh.md)

最新SA7–SA10：四个新开发案例真实Flash完整契约4/4；三模态小规模真实数据模型均完成交付（数值/图像在修正实验评分器误拒后恢复，非首次无人工3/3）。173项相关回归及三个冻结模型在当前源码新进程预测通过。该自动推进窗口共12fit、10API、课程¥0.080768。[结果与边界](docs/batches/SA9_SA10_delivery_closure_20261126_zh.md) · [后续自动队列](docs/AUTONOMOUS_NEXT_QUEUE_20261126_zh.md)

本分支`codex/ab-semantic-autonomy-1126`已将A的SA1–SA3合入B最新完整研发版`b0944ae`（个人发布`c6ea0b7`的同源源码）。当前入口是本分支根目录；新增`--task-interpretation proactive`可与B的`--task-analysis`和自审组合。最初SA4集成验收150项通过、两条真实worker烟测共4fit/0API；后续进度见上方最新记录。下文课程分数属于B原版本，本集成版尚未重新评分，也尚未最终冻结。[共同交接与复跑说明](docs/AB_SEMANTIC_AUTONOMY_HANDOFF_20261126_zh.md)

## 队友的 Task1 研发起点（11 月 26 日）

以下为B来源 `codex/task-driven-planning-1125` 的研发起点记录；共同继续开发请使用上方的A/B集成分支。B按老师最新自主性要求核验后，在其同一份源码上重新冷启动，Task1 指定验证 **1,533/2,030（75.52%）**，Task2 **684/940（72.77%）**，各 10 次拟合、零模型 API。旧版 Task2 的 73.30% 有独立的运行记录。交接页包含两版结论、旧版输入特例、启动命令和通用表格候选的接入位置。[两版核验与 Task1 开发交接](docs/TASK1_DEVELOPMENT_HANDOFF_20261126_zh.md)

## 当前统一入口

统一入口会保存题面声明的字段顺序和类别含义，按训练标签解释指定类别的召回率，并在生成候选和执行训练时遵守任务声明的方法限制。[任务契约修复、核验方法与日志位置](docs/TASK_CONTRACT_EVIDENCE_20261126_zh.md)

早期自主基线 `d501807` 已封存，指定验证成绩为 Task1 **1533/2030（75.52%）**、Task2 **684/940（72.77%）**。本次任务契约修复以分支根目录的最新版交接。研究分支可通过 `--self-review` 检验 Agent 提出的规划改动。四组同预算对照未显示提分，默认沿用验收过的规划器。[自主验收与实验结果](docs/RSI_PLANNER_1125_zh.md)

本地研发可将每个任务的总训练上限提高到 16 次，运行时使用 `--fit-limit 16`。配合 `--protected-self-review`，Agent 可先探索、再回头复核；当它准备停止且还有余量时，会先保存原模型，再让一个合规候选在两份训练内划分上挑战。两种路线共用同一个 16 次上限。[9 次与 16 次对照](docs/RSI_BUDGET_1125_zh.md)

从仓库根目录运行。给定训练任务和待预测输入，Agent 会读取任务声明、在训练集内选择或组合方法、重训并生成预测。下列命令也适用于已解压的任务目录。

```sh
python -m unified_agent solve TASK.zip --prediction-input FEATURES.zip --run-dir runs/new-task --output runs/new-task/predictions.csv
```

也可以先运行 `python -m unified_agent train TASK.zip --run-dir runs/new-task`，待模型封存后运行 `python -m unified_agent predict FEATURES.zip --run-dir runs/new-task --output predictions.csv`。运行目录须为新目录。默认由本地规划器依据训练反馈选动作，每次任务最多 10 次实际分类器拟合，包含最终重训。机器上已登记的编码器和已安装的分类器向所有兼容任务开放。每轮候选、实际动作及其依据写入 `agent/decisions.json`。

原始表格可用 CSV、TSV 或 Parquet。任务说明可用 `Target column is verdict. Primary metric is macro F1.` 声明标签和指标。共用一张表时，可用 `Split column is phase.` 声明行划分，训练只接收标记为训练的行。图片可以按 `train/类别/图片` 或 `类别/training/图片` 摆放。运行支持压缩包、已解压的目录及单个表格文件。任务包有明确清单时优先读取清单。

若特征无法解析，程序会核对是否仍有可信训练标签。根据这些标签生成的多数类预测在运行记录和预测凭据中均标为 `partial`。类别依据缺失时写入 `unavailable` 及原因。默认预测文件包含顺序行号和分类结果，ROC-AUC 任务另含各类概率。详见[自主决策、格式与交付验收](docs/AUTONOMY_AND_UNSEEN_INPUT_20261125_zh.md)。

输入和结果的可核验边界见[自主决策与陌生任务格式验收](docs/ANONYMOUS_PLANNER_AND_INPUT_VERIFICATION_20261125_zh.md)。旧方案可使用 `train --policy legacy` 复现。[旧版环境与成绩](docs/UNIFIED_AGENT_20261124_zh.md)

## 历史四次搜索对照

该阶段使用本地 `interleaved_routes`、四次搜索、前三名三划分复核、全训练集重训。两个发布任务使用相同配置。该阶段的发布验证、本地复验和实验协议见下列文档。此节之后的 v0.42 结果对应仓库随附的历史模型和报告。

- [预算复验和当前发布验证](docs/search_budget_replication_results_zh.md)
- [搜索分数反馈与实际复核耗时的配对协议](docs/current_stage_10h_zh.md)
- [搜索反馈与复核开销的实测结果](docs/feedback_value_results_zh.md)
- [历史决策证据复核](docs/decision_evidence_audit_zh.md)

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
