# A-19：将任务包明示目标带入契约（2026-11-21）

预登记0训练/0API。A18发现Task1/Task2旧适配器只读训练字段，丢失包内README的明确metric/标签语义；真实原文确认两任务要求report test accuracy，Task2明确1表示headline_A点击率更高、0表示B。训练/验证/正式test数据均未因此新增读取。使用README内容恢复目标，不按任务名路由算法。

A新增保守的只读任务说明适配：只接收唯一README.md/txt，忽略__MACOSX；记录文件/字节摘要、规则命中与短原文。只识别明确的正向report test accuracy声明；否定/冲突/多个说明保持未知。pair规则限定已加载headline_A/B、label_pairwise格式，并识别直接的1 if A ... 0 if B声明，形成preference/left契约；改写、反向标签、否定或歧义不得猜测。原manifest已有schema优先，不靠README覆盖声明。

接到load_training_task与legacy development两入口，goal_sources新增declared_task_readme来源，保持未命中仍是unknown。README只是数据，不执行其中代码/指令。每次提取保留来源供模型与日志解释。规则有限，未命中并不意味着任务没有要求。

软件验收：正向声明保留、否定/多README/反向偏好不确认、两加载入口目标一致、原manifest优先、不读取val/test成员；真实Task1/2训练包重建A18就绪矩阵确认目录可用，不训练。包括原有目标/输入/AB边界无实训回归。A17 Q使用此前源码并已封存，作为原版本开发证据，不伪装新版本最终Q。
