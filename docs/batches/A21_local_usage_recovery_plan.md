# A-21：不等待学校平台的本机历史用量审查（2026-11-21）

用户明确无法调出平台记录、要求继续推进。旧11条unknown状态保持不伪造；不新建空真实账本，不把fixture账本用于真实请求。不因人工对账不可得而停掉本地实验。

本批0fit/0请求，审查flash_history_unresolved.json指向的真实usage/events及已有audit来源，排除已明确fixture运行。每条记录源文件SHA256、request指纹、请求模型、messages字节数（不导出原文）、已知输出上限/完整usage/HTTP状态等。若能恢复实际token，必须有同一请求身份的原始usage证据；不能按时间相近猜测或将无响应当0费。

若只能求上界，标记upper_bound而非真实usage；只有已记录且适用的请求输出上限、输入字节保守界、模型价格身份和完整请求清单都可核验时，才考虑单独预算区间方案。缺输出上限或记录不完整时明确标unbounded，不用拍脑袋估数。先输出可审核结果，再决定真实API接续，不边审查边发请求。

runner scripts/audit_a21_local_unknown_usage.py，输出protocol/results/completion。不得输出API Key或请求原始训练内容；只摘要/指纹。不得修改历史原日志或把其unknown改known0。用户不需要再重复查询学校记录。
