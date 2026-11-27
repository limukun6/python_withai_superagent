# SA2：LLM输出提案可编译为真实提交文件，73项离线检查通过

2026-11-26，基线SA1 40be241；同分支codex/a-semantic-adaptation-1126。先计划、代码、runner，再运行。本批0实际分类器fit/0真实API/¥0。回执73项通过、运行记录11.031秒；包含一次独立CLI预测进程，不含人工开发时间。

## 已接进什么结构

semantic-task-v2可同时解释任务和提交要求，新增output-csv-v1 recipe；SA1的v1仍兼容。不增加一次独立必需LLM请求：任务解释的同一JSON可包含输出计划与真实公告引用，错误仍由SA1试读修复循环处理。

新增output_recipe.py，真实predict与partial_classification调用它；契约冻结在task_spec，并沿既有task_spec_sha256保护。LLM只描述字段来源、映射与顺序，不能提供预测数值。程序使用模型输出、原始ID或明确的顺序编号来写文件，重新读回核对后原子发布；验证失败保留原文件。

| 瓶颈 | 结构 | 实验观察 | 支持边界 |
|---|---|---|---|
| CSV表头和顺序ID写死 | 公告提案的columns/encoding/delimiter/header | Ticket/Answer、UTF-8 BOM、分号、无表头TSV、一基索引可按要求导出 | 当前只实现CSV方言，不是任意提交容器 |
| 原ID前导零丢失，按位置乱对齐 | 从冻结record reader提取原ID；按样例ID连接重排 | 0001/0010保持；sample顺序0010/0001时标签也正确随行重排 | 不给LLM生成或猜ID；重复实体ID可在明确unique=false且原行序下保留，模板重排需唯一行ID |
| 正类概率与模型列序混淆 | 输出列绑定训练class_label，查真实classes_定位概率 | 主指标macro-F1仍可导出accept概率；模型列序decline/accept，导出正确0.9/0.2 | 没有改变训练目标；不是模型质量收益 |
| 选出的模型可能不具备提交接口 | 按output能力过滤实际caps并记原因 | 官方循环过滤label-only、保留predict_proba，指标仍macro-F1 | 不改B排名算法，仅准入约束；未新增候选或训练 |
| partial兜底丢提交格式 | prior与已验模型走同一输出recipe | prior保留真实ID、输出0.5类别先验且仍标partial/refit=false | partial不计完整自主成功 |
| 看似写出文件但字段/值错误 | 独立重读检查＋原子发布前验证 | 坏模板、ID集合不符、缺ID、重复ID、标签/概率被篡改均拒绝；旧文件保留 | 检查序列化与来源一致；不能自动证明LLM理解公告正确 |

## 验收与资源

新增tests/test_output_recipe.py 18项；与SA1及既有格式/自主策略/指标/恢复测试合计73项通过。runner显式阻止Executor.worker及网络连接。预测模型是无fit的固定算术夹具（9→accept、2→decline及固定概率），不是已训练分类模型。用真实joblib、create_predictions和新进程`python -m unified_agent predict`导出，模板重排后的CSV逐字节一致。

原无output_recipe的路径仍写原row_id/prediction；已有相关测试未改。events.atomic_text增加可选编码与发布前检查回调，旧默认UTF-8保持。真实LLM尚未参与，因此不报告“LLM能正确理解任意输出说明”。本批没有读取/重评课程正式测试，没有改真实账本。

证据：research/sa2_output_recipe/{protocol,tests.log,results,completion}。protocol绑定六生产模块、测试和runner SHA，测试前后不变。

```powershell
Set-Location -LiteralPath 'D:\codex\python_withai\runs\semantic_adaptation'
& 'D:\codex\python_withai\runs\a79_encoder_assets\venv\Scripts\python.exe' -m scripts.run_sa2_output_recipe --output research/sa2_output_recipe_repeat
```

## 当前仍需做的差量

1. **SA3：显式解释模式和探查工具循环。** 普通native快路成功时仍不会主动让LLM审查陌生输出公告；不能称全入口输出要求已覆盖。当前新recipe主要来自SA1语义fallback编译，下一批应支持解释模式选择/主动输出契约解析。
2. **结构证据与预测前试读。** recipe可在训练时通过schema检查，真实预测ID/样例只在预测阶段核对，尚未实现预测前主动工具检查与语义修复；应减少训练完才发现缺ID的情况。
3. **字段改名/封装变化。** 仍需把实际预测字段映射到冻结模型输入，不让“名字不同”直接变成无法交付。
4. **更多输出容器/读取器。** JSON/NPY等输出、非record输入原ID提取、跨文件样例定位与非唯一ID模板连接未实现，不应按“任意格式”宣传。
5. **真实Flash与真实原包无人干预对照。** 先冻结独立保留案例和调用预算，再运行；现有测试只是开发工程证据。

项目继续结构研发，尚未冻结。SA1/SA2解决的是可执行解释/导出接口；完整工具推理与真实模型语义效果需要后续证据。
