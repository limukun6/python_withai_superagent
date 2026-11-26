# SA1：全局语义解释已接入真实入口，离线43项通过

2026-11-26。基线B d501807，分支codex/a-semantic-adaptation-1126，目录D:/codex/python_withai/runs/semantic_adaptation。先保存计划和runner，再执行。本批真实分类器fit=0、真实模型API=0、费用=0；43项检查3.219秒，不含人工开发时间。

## 瓶颈→结构→实验→证据边界

| 瓶颈 | 改动 | 验证 | 边界 |
|---|---|---|---|
| 多说明与非标准源无法解释 | semantic_contract.evidence_workspace联合真实文档块和成员索引 | 两份非标准名markdown、无train/test文件名的表，原adapt_training拒绝，新入口成功 | 模拟模型选契约，不是真Flash理解成功率 |
| 跨句同义目标先被词面规则卡住 | 模型直接提出字段/目标/指标/特征与引用，不经过GOAL/相同短语验证器 | 原目标不是label/y/class，编译为8行字符串标签、macro-F1、数值特征 | 只验证证据ID真实和契约能执行，**不能机器证明语义解释正确** |
| 一次错误计划就退出 | compile_plan真实试读→具体诊断→最多一次语义修复 | 首提案target缺失，第二提案修正成功；超额不第三次调用；API异常不重发 | 均为脚本客户端，无真实修复调用 |
| 训练/预测读法不一致 | 复用既有input_recipe/predict_records冻结读法 | 同包分离训练行，验证行标签不进入任务/提示词；两查询行按[9,2]重放 | 未训练/导出模型，不称端到端分类交付 |
| 原有正确入口被新层扰动 | native→原adaptive→新semantic fallback，标准快路0模型调用 | 官方_train实际进入新分支并到standby，Executor.worker被拦截、无fit | 尚无LLM-first开关；默认无配置客户端时仍本地 |

新增superagent/semantic_contract.py，修改autonomous_run的任务读取接线；B菜单和排序不变。对已知评价成员、目标/分组列进入特征、伪造引用、未解决不确定性有拒绝检查；有效显式manifest由原契约读法处理，本批不让模型覆盖它。

## 测试和保留失败

新13项模拟客户端/真实解码检查，连同既有adaptive_formats、anonymous_policy、task_readme，共43项通过；runner禁止Executor.worker和socket.create_connection。测试给官方入口在训练前设检查点哨兵，不发生真实fit。protocol记录五份改动/测试/runner SHA，运行前后不变，见research/sa1_semantic_contract。

初次相关31项回归出现两项Windows测试夹具问题：SQLite连接with事务结束后未close，清理报WinError32；含§的fixture按本地cp936写出，却由生产UTF-8读入。只修夹具的closing/显式UTF-8，未改变生产SQLite或编码解码器，未放宽断言。首失败见本次工具执行记录；没有生成原始失败log文件，本文只记摘要，不伪造原回执。

复跑：

```powershell
Set-Location -LiteralPath 'D:\codex\python_withai\runs\semantic_adaptation'
& 'D:\codex\python_withai\runs\a79_encoder_assets\venv\Scripts\python.exe' -m scripts.run_sa1_semantic_contract --output research/sa1_semantic_contract_repeat
```

## 下一批差量：SA2，而不是冻结

1. 公告驱动OutputRecipe与独立提交格式校验；现有CSV固定表头还没改。
2. 结构化探查工具循环和显式LLM-first/辅助模式，避免只在异常后才注意到错误默认值。
3. 复杂/非常规manifest解释的正规化：保留权威声明，不因reader只支持旧schema就拒绝；不能用新层悄悄覆盖明确目标。
4. 预测列映射、任务类型/偏好语义、split协议等扩展：本批主要编译标量表，未实现所有适配类型。
5. 冻结独立保留案例后做真实Flash探测。本批案例只用于开发；历史API账本和新B待并账调用先核准，不能新建空账本。

限制：文档目前接受markdown/rst及既有标准txt文档名，有字节上限；长文档/任意扩展名/PDF尚无通用文档工具。解码仍用已实现reader；不生成代码，不新增评分公式，不让LLM输出分类结果。新路径存在不代表任何陌生任务都能成功。
