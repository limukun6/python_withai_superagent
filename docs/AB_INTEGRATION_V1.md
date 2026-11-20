# A/B 共同接入约定 v1

2026-11-21增量：A已补目标来源/metric_spec身份与导出校验；提供scheduler_bridge的动作校验及注入式分发（尚未接正式CLI）。新交接见A_TO_B_NEXT_HANDOFF_20261121.md，实验结果见batches/A_window_20261121_results.md。接口字段继续沿用本文v1。

双方分工：我们 A 维护任务接入、标签/指标契约、事实核对、统一 CLI、执行日志与模型导出；B 维护方法库、强基线和 explore/confirm/stop 的调度。两人都实现与运行实验。该文档、examples/ab_v1/input.json 和 tools/export_b_handoff.py 是交接入口。

## 共同基线与提交

交接基线是包含本文件的 GitHub `codex/expanded-feedback` 提交；发布后的精确 SHA 由交接消息给出。该分支基于 39c0522，包含 A1-01 至 A1-04 的实现、测试、运行证据和 A2 prepare 脚本。B 的既有强方法仍在队友分支，应在该基线上适配其模块，或用此提交作为 A 线合并来源，不直接覆盖另一方 checkout。不要把远端 main、旧交付包和本机未提交代码当作相同版本。

目前已实现：raw-history/checked-history 正式 CLI，共同 DecisionState/ExperimentProposal 接口，实际分类器核对、失败记忆、受限解释核验，固定三划分复核、refit、冻结预测与累计账本。

仍待合并：B 的强候选目录（包括第6次之后才可触达的旧强方法）、预算 Scheduler 与正式循环的动作分发。当前 CLI 仅接受既有固定 allocation，不能说 B 的调度已接通。B 只需提供一个统一可执行候选目录与 Scheduler 适配器，A 负责共用循环接入；运行前再次核验候选能力和预算。

## 输入接口

`ab_schema_version=1`：

- task：TaskSpec.record()，含 kind、主指标、类别、input_schema；任务名与文件名不作方法选择依据。
- history：已完成或失败的 trial、候选配置、主指标/辅助指标、实际 estimator 与核对。未知字段明确为 unknown/null。
- available_candidates：两种规划器完全相同的可执行候选集合。强方法按通用类型/资源能力进入目录，不依赖先跑满6次或特定任务名；两组预先暴露相同元数据。未选中不训练。
- prediction_evidence：按 trial 提供逐行预测、run-local 训练索引、split/evaluation-payload/label/prediction 指纹及证据来源。不同 split 的向量不能直接逐行比较。原始标签不发给 LLM；B 的可信本地评价代码从执行器读对应内部数据，不访问外层或正式测试。
- budget：fit_limit=11，attempted_classifier_fits、completed_classifier_fits、remaining_fit_allowance、reserved_final_refit=1，llm_request_limit=3、used_llm_requests，以及墙钟/每次训练时限。

实际 DecisionState 目前不含预测向量和完整资源账本；适配器导出补齐这些字段供 B 的可信代码使用，不将大型向量原样塞入 Flash 提示。示例只使用匿名合成小任务、去除候选/算法选择之外的任务身份。

可从一条已有真实 run 导出给 B 的 JSON：

```powershell
# 在本分支根目录运行，run-dir 指完整或部分已执行的研究目录
python -m tools.export_b_handoff --run-dir PATH_TO_RUN --output NEW_HANDOFF.json
```

导出器保留原始 trial 与记录的性能预期/核对，不自动宣布模型好坏。可按 split 指纹取逐行配对证据；每次动作仍需经过执行器的合法性检查。

## 返回接口

```json
{"action":"explore","candidate_id":"numeric_tabular_logistic_C1.0","split_id":"primary","reason":"Try a different configuration","evidence_trials":[1],"estimated_classifier_fits":1}
```

action 三选一：explore 表示未尝试候选，confirm 表示已有完成候选在新的预登记划分上复核，stop 表示结束搜索并交给共用选型/refit。stop 的 candidate_id/split_id 可为 null；它不是让调度器人工指定赢家。reason 与 evidence_trials 必须能对应已发生观测。estimated_classifier_fits 是 B 的预算声明，A 按实际训练账本核对。多组件模型、OOF和最终融合训练都计入11次。

首次接入 B 可以直接包装现有方法对象：`Candidate(family, params, fit_predict)`；fit_predict 接收训练输入、训练标签、待预测输入，返回模型与预测。模型同时提供统一 predict/probability 适配；缺概率的模型不能用于要求分数的指标。候选还需补齐语义、输入支持、内部拟合次数与资源元数据；recipe 不得因缺少 swap flag 被当成安全对称模型。

## 公平实验与验收

1. 同 task/split 的本地与 Flash 组使用同一目录、索引和实际训练11次上限（含refit）；Flash每路径最多3请求。确认不额外调用LLM。算法内部多次fit按真实次数计账。
2. 双方共用每任务累计真实账本；模拟账本不可导入。费用口径沿用当前仓库已冻结的空闲时段费率说明，不在本接口另改费率或宣称新核验。未知历史费用保留未知。
3. 先跑匿名任务：A入口产生状态→B返回动作→A执行/反馈→B继续→A选型/refit/导出。日志记录实际输入摘要/指纹、动作理由、证据与消耗；模型在新进程重载预测逐行一致，再启动大矩阵。
4. 机器时限用首轮小试校准，两组采用共同上限和相同缓存规则；轮换顺序。不同任务可有不同能力适配，不手选任务专属赢家。
5. 大批前冻结动作字段、候选与各自文件归属。A主改 cli/benchmark/contracts/data/predict/反馈模块；B主改方法库及scheduler。修改共用控制循环前用提交记录交接，不同时写同一文件。

## 当前真实进度与未决项

A1 最终入口验收5项通过，宽回归141项通过；真实API为0。A2已用四个真实任务生成12个冻结状态，12次拟合、50.225秒，无validation/test访问、API=0。A2状态通过当前源码内容哈希绑定；B合并若改superagent源码，旧状态作为历史证据，新版需重新冻结或单独标记兼容性，不冒称同版本结果。

用户已确认11条历史未知请求尚未对账，含共享诊断。本机只找到模拟SQLite账本。真实付费调用暂停，状态准备、方法合并、匿名本地验收继续。请勿新建空账本掩盖旧请求；B若有完整真实账本，可提供路径/文件指纹按既有规则核对。

给B的第一项交付：在共同基线上把强方法放入可执行候选全集，提供一个匿名任务下 Scheduler 返回 explore/confirm/stop 的日志与模块提交。A随后接入共用动作分发，双方各跑完整子矩阵。
