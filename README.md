# AIAA 3102 · General Algorithm SuperAgent

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
