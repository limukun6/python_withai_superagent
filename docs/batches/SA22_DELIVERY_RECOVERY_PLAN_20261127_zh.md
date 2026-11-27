# SA22：训练后新到数据的预测恢复，不重训

基线5f820bc。A负责unified_agent、predict及新增delivery_recovery；B目录/排序不改。先保存计划和复跑代码，再检查/实跑。

## 结构与边界

当前create_predictions输入或输出来源不吻合就退出，模型虽在但没有恢复路线。新建预测适配异常分类；正式predict/solve接到恢复控制器，由LLM查新包结构、提出data-only预测适配，再实际重放特征与输出预检、复用原模型导出。

只允许更换预测成员/集合/分隔符/字段映射、已支持的数组或图像预测描述符，以及既有input_id列的数据来源字段。不改目标、类别、指标、训练列/列顺序、预处理、输出列名/编码/分隔符/标签映射/概率类别，不把新格式问题变成新训练。适配器绑定模型、原task_spec、查询包及reader代码SHA，独立新进程可0API复用；原模型与task_spec不修改。

training_protocol补任务包路径/SHA和学校账本scope/path；预测修复必须同scope同账本，合计最多3请求，不因换进程重置额度。旧模型正常predict保留；缺绑定的旧模型不凭空新建付费scope。没有剩余额度则保留模型与诊断，不能承诺所有阶段都能继续付费。仅数据接入/输出来源错误进入LLM；模型损坏、磁盘/权限、真实训练失败不自动重训或重试HTTP。

## 验收设计

先无fit/API回归：可改项/冻结语义、哈希绑定、同账本scope累计请求、缓存0API、错误分类及异常不丢模型。模拟检查与真实交付分开。

两条正式路径D1/D2使用SA21既有语义解释显式重验证迁移（0新API），各从统一train自主训练一个新模型（最多2fit含refit），而不是给旧训练scope另开额度。训练与后续predict用本批同一新scope，训练0API。随后收到与既有预测特征数值相同、布局不同的新包：

- D1：原JSONL改为另一成员名/字段名和ID字段的CSV，原输出tab/BOM/概率类B/原ID含义不变；新包说明明确对应关系。
- D2：原features.npy改为一个新NPZ成员及新键，原分号/零起索引输出不变。

两个都是受控的新到格式条件，非新的独立来源或未知任务性能；不注入坏模型回复/执行异常。先0API预测原包得到本模型参考输出，再让正式predict面对新包自然失败、自主恢复；第三个独立进程复用适配器预测，须与参考CSV逐字节一致。核对模型、task_spec、selection和fit账本的前后SHA，恢复新增fit必须0。

每路径整个新train＋predict最多2次实际fit、3次Flash、360秒；两条合计最多4fit/6API，估计停止阈值各¥0.30、总¥0.60。预计3–6分钟。共享canonical真实账本，510条/16旧未知/0pending核对后运行；旧未知保留null/SHA，新增unknown/pending立即停当批、不重复失败HTTP。

主指标：恢复到正确完整交付、错误完成、恢复新增fit、人工介入、跨进程/重复predict的新增API、总耗时/fit/API/token/课程折算。学校实际扣费/历史预算合规仍未核验。失败保留，新目录protocol/results/completion；密钥、模型、原包、SQLite不进Git。

脚本scripts/run_sa22_delivery_checks.py、scripts/run_sa22_delivery_live.py及审计。若遇无法执行的原生任务/模型计算错误，本批如实保留边界，不扩成任意代码生成或无限重试。训练0API源于显式解释缓存，不能伪称一条全新原任务从读题到交付只花本批预测费用。
