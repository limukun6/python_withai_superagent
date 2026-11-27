# SA15：默认入口接管通过，旧真实回复可显式重验证

基线e3aaee4。A只改语义接管和证据迁移，不改B方法/排名。

| 瓶颈 | 结构 | 实验与证据 |
|---|---|---|
| 看见.npy就当成已有权威契约，无manifest也不让LLM接手 | 默认fallback区分“专用reader认领”与“显式manifest”；仅前者允许按同一共享预算进入全局解释 | 7项新检查，含真实NativeContract分支、blocked/耗尽/无client不请求、显式manifest不覆盖、本地成功不请求、无文档不猜、正式_train到fit前检查点 |
| 新reader无法复用旧native真实提案 | 显式工具重新编译semantic-native-v1；旧artifact仍拒绝直接跨源码使用 | SA14两条真实提案显式迁移，原artifact不改，9项契约各全部通过；两旧模型54行新进程预测逐字节一致，0新fit/API |
| 最后一次API可能是B规划，不能当原读题来源 | 找语义接受之前对应的request/response，按ID选usage，核对原包SHA、proposal、model/token/费用和canonical已结算行 | 9项新检查；本轮真实D1在后续B规划调用之后再迁移，确认选中第二次语义请求，未串用第三次B请求 |

完整相关回归**261/261、13.484秒、0fit/0API**。含16新＋原245项；两条旧Dummy.fit夹具仍显式排除。旧receipt没有包绑定时保留legacy缺证标记，不伪造原始包证明；SHA不是供应商签名。

## 默认CLI真实验收

使用SA14 Iris原96训练/24预测行改为非标准observations.npy/decisions.npy/arrivals.npy，原文写明balanced_accuracy、分号/BOM与行序；不提供manifest。是已有开发数据的重封装，不能当独立新任务。

从standalone `unified_agent solve`启动，**不加--task-interpretation或--task-analysis**，stdin关闭，无client/resolver/算法注入。真实日志首先出现semantic_reader_handoff，随后模型给出native契约。第一份提案的所有结构已正确，却把“没有正类”写进uncertainties；三分类balanced_accuracy不需要正类，程序仍按旧flat uncertainty规则拒绝。第二次提案保持相同数据/输出含义、清空疑问后通过。该多余修复真实计费，不回填为首次成功。

第三次调用是B的experiment_planner，最终自主选RandomForest，实际**2fit含refit**。24行完整输出，9项语义检查全部通过，新进程预测与原CSV逐字节一致。未打开预测答案或新评分。

| 角色 | 输入token | 输出token | 课程费 |
|---|---:|---:|---:|
| 初次语义提案 | 1933 | 1242 | ¥0.006901 |
| 语义修复 | 1921 | 919 | ¥0.005597 |
| B模型决策 | 6029 | 293 | ¥0.007201 |
| 合计 | 9883 | 2454 | ¥0.019699 |

真实批次82.938秒、3API/2fit；语义2次/¥.012498与B规划1次/¥.007201分开，不将全部收益/费用归A。两次回复迁移和审计均0新fit/API。账本477→480，known¥8.083541→¥8.103240，原16未知/旧行SHA不变，pending0/new unknown0；实际扣费/全团队总额仍未核验。本自动窗口SA7–SA15累计24fit/19API/¥.136845。

## 下一缺口

这次解决了“读不出来却不让模型接手”。**默认fallback在本地能读出输入时仍可能跳过LLM，因而漏掉文档中的自定义输出要求**。下一批应分别验证本地读取成功与任务语义完成，再决定人用CLI的自动模式；旧fallback须保留用于本地对照/兼容。另有flat uncertainties把可选信息当阻断项的问题，需要有作用域的说明或契约，不能直接忽略任意自由文本疑问。

未冻结、不保证任意未知任务。程序化API的默认行为与人用CLI须明确区分。每批先计划/代码/协议，不能为了证明泛化连续重跑同题到成功。

## 交接文件

- 计划docs/batches/SA15_default_handoff_plan_20261127_zh.md。
- scripts/run_sa15_default_checks.py；research/sa15_default_checks_01。
- scripts/prepare_sa15_default_case.py；research/sa15_case_prepare_01。
- scripts/run_sa15_receipt_revalidation.py；research/sa15_receipt_revalidation_01。
- scripts/run_sa15_default_live.py；research/sa15_default_preflight_01、sa15_default_live_01。
- scripts/audit_sa15_default_live.py；research/sa15_default_audit_01。
- 原数据/模型/快照留忽略的runs/sa15_cases、sa15_default_live_01等，不进Git。已尝试scope不能原样重跑。
