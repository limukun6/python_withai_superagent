# 让新安装的分类器自动进入共同目录

使用这一版的 `--method-space runtime` 后，Agent 会发现 sklearn 提供的分类器，也会读取已安装 Python 包通过 `superagent.classifiers.v1` 公布的分类器。开发者用标准包元数据说明入口一次。任务启动后，Agent 自行判断适配性、生成参数、安排试验和决定最终模型。完整补丁可直接应用于共同提交 `9779944d3467b9df8379101355179cb9916d5d84`。它已包含前一版共享菜单改动。[完整补丁](current-team.patch) · [可直接解压的运行源码](runtime-source.zip)

扩展需要提供可默认构造、使用 CPU 的 sklearn 兼容分类器，输入为二维特征，每次训练消耗一次监督拟合。Agent 会读取类别数、稀疏输入支持和概率输出等能力。多组件训练需要独立计数。扩展的包名、版本、入口别名及实现目标会写入配置，交付的模型会在新进程重新检查所需版本。任务数据中的字符串不能指定要导入的包。

包作者可在已安装包的 `pyproject.toml` 中注册自己的实现。例如下面的包会公布两种不同能力的方法。[教学分类器源码](provider_fixture.py)

```toml
[project.entry-points."superagent.classifiers.v1"]
median = "independent_methods:MedianPrototypeClassifier"
threshold = "independent_methods:BinaryThresholdClassifier"
```

使用含所需依赖和扩展的同一 Python 环境训练与预测。独立源码包里有共同入口及实验脚本。

```bash
python -m superagent run train.zip --planner local --scheduler ab-v1 --portfolio ab-v1 --task-profile train --method-space runtime --fit-limit 11 --run-dir trial
python -m superagent predict features.zip --run-dir trial --output predictions/rows.csv
```

## 这次分别验了什么

匿名两分类训练的三次探索中，两种新方法每次均在候选中。Agent 实际试了极端随机树、逻辑回归、支持向量机，再选极端随机树重训，共 4 次真实拟合。两种新方法均没有被排上训练。换为三分类或要求概率的输入契约时，Agent 自动排除只能输出二类标签的扩展。模拟缺失的可选入口被单独记录，其他方法正常工作。[原生动作](actions.json) · [实际可选方法与训练](result.json) · [能力筛选](contracts.json)

另用正式训练 worker 指定中位数原型分类器做了一次接口验收，计 1 次真实拟合。两个独立的正式预测进程导出相同的 33 行结果；将扩展版本由 1.0 换为 2.0 后，旧模型在导出前拒绝加载。这次指定训练仅计入工程验收。第一次整批启动暴露出第三方实现无需声明 sklearn 私有参数表，发生在拟合与调用之前。修复后按事前方案完成了同一个匿名夹具。本次 5 次实际拟合，模型 API 0 次。这里没有任务准确率评测。[事前方案](protocol.md) · [首次启动记录](first_attempt.json) · [独立训练计数](interface_physical.jsonl)
