# A线下一轮：可核验决策、纠错闭环与按证据更新的短程计划

执行更新：A145–A147已完成，新增8fit/12Flash；当前variant不展开A148、默认不改。实际结果和局限见[batches/A145_A147_grounded_protocol_results_20261124_zh.md](batches/A145_A147_grounded_protocol_results_20261124_zh.md)。以下为开跑前方案与预算快照，不能当成已消耗或当前计数。

设计基线：fb791d5，94个生产Python源。此文是下一轮方案，尚未实施或运行。当前A累计1636fit；真实唯一ledger375/known¥6.8545065/unknown12/pending0，A持有空闲。原12未知按用户授权保留，不等待官网账单；新增unknown/pending仍停止当前付费批、失败不自动重试。H2/Released2由B继续，本文不要求其补交已有材料。

## 1. 判断：A线应该突破哪里

当前目标/指标契约、合法目录、fit计数和独立交付已形成可用基础；下一轮主攻“为什么这样决定能被核实”和“发现错误后下一步是否真的修复”。不把新增分类算法、候选排序或探索/复核预算重新包装成A贡献。

| 当前实证 | 暴露的具体瓶颈 | 对应下一步 |
|---|---|---|
| 参数语义40开发例36符合；旧4条明确C错误只检出1条 | 自然语言否定、参照和多数字使检查器大量unknown；继续补短语规则难保证覆盖 | 决策输出绑定结构化事实和显式参照 |
| 最新Banknote的3份解释产生34条prose检查记录，16条unknown | 机器能执行合法候选，不等于理解或解释都正确。检查记录不是独立语义命题，也不能直接算自然错误率 | 将可核实依据与自由假设分离，报告覆盖率 |
| ab_run在训练后附explanation_audit；memory保存warnings，但没有问题关闭状态 | 目前主要是“显示检查”，缺少问题→修复要求→新证据关闭的闭环 | 有作用域的问题记录和决策门 |
| A140紧凑反馈少6.15%输入token，三任务下一候选均相同 | 压缩改善输入开销，尚未改善动作或纠错 | 保留压缩基础，下一轮测决策依据/修复行为 |
| Banknote本地15.562秒/Flash155.312秒，同280/280；API等待140.766秒 | 每次新探索都调用LLM不一定值得，等待是主要开销 | 条件失效时重规划的短程计划，减少无效委派 |
| Banknote4250输出token中3207为reasoning token（本次usage） | 让可见回复短一些不等于能稳定减少推理/延迟 | 统计调用数、总tokens与时间；不承诺提示压缩等比例加速 |
| A141可恢复partial或完整模型，显式不支持续训 | 恢复模型交付仍不等于恢复实验进度 | 条件扩展：有预算继承、动作幂等的安全续跑 |

H1的Flash增益来自一个对称任务的选择路径；Banknote无增益。两者说明任务依赖，不能据此按数据集名字路由或认定所有任务都该关闭LLM。Banknote已看过外层成绩，此后只作已见开发/效率案例，不再作为新修改的未见评价。

## 2. 优先级P0：把已有事实检查接成可核验决策协议

### 已有模块与新增差量

复用TaskSpec、MetricSpec、measurement_claims、configuration_facts、comparison_scope、estimator_facts。已有这些模块，不声称从零发明事实核验。新增的是：统一事实身份、让Flash在正式输出中引用/组合事实、执行前检查这些引用，再用已核实依据生成执行记录。自由解释仍可保留，但不能升格成已验证事实。

事实最小身份：run/task契约SHA、catalog SHA、trial、split_id、阶段train/search、metric及类序、payload/row/label SHA、记录来源与版本。不能只用“trial2 accuracy”跨运行复用。记录类型分开：observed（真实测量/实际对象）、declared（待执行配置）、definition（有来源的参数含义）、hypothesis（未来猜想）。只有实际匹配的证据才能确认相应类型的断言。

提案示意（是设计，未实现）：

```json
{
  "candidate_id": "某个当前合法候选",
  "purpose": "improve | coverage | diagnose",
  "evidence_refs": ["当前run/primary/trial2/search/accuracy"],
  "comparisons": [{"left": "factA", "op": ">", "right": "factB"}],
  "parameter_relation": {"reference_trial": 1, "parameter": "C", "direction": "weaker_regularization"},
  "addresses_issue_ids": [],
  "hypothesis": "尚待实验检验的预期",
  "uncertainty": "目前不能确定的部分"
}
```

实际字段必须按purpose条件化，不能强迫所有候选填C或强迫coverage预测涨分。improve引用合法测量及候选能力；coverage引用尚未试过的合法路线；diagnose引用具体待解决问题。历史尚空时不强造trial引用。声明中的未来候选不能冒充已经拟合的对象。

关键检查：同split/metric/类序及受控行号的测量才可比较；同一有效参数化内才比较C方向；C增大不自动推出提分；一次内部涨分不关闭因果不确定性。实际不支持的字段/缺证据继续unknown，不能为了覆盖率全部认证。

建议新增fact_registry.py、grounded_proposal.py，在ab_flash的context_for/validate_proposal接可选grounded-v1。默认旧协议不变，两组共享全部候选、训练预算和B调度。TaskSpec仍保留unknown意图；LLM不能把未声明正类或指标变成用户真实要求。

## 3. 优先级P0：从warnings变为有关闭条件的问题记录

新增IssueRecord，而不是再加一个自由文本Reviewer：

```text
issue_id, code, severity, task/run/source_version,
trial/split/metric/candidate/claim_scope,
observed_fact_ids, contradicted_claim,
permitted_resolution, required_postcondition,
status = open | addressed_pending_check | resolved | superseded | unknown
```

例：错误C方向只能绑定对应估计器、同路线参照和参数关系，不能标记“整个算法无效”；训练超时不能变成方法质量失败；旧split的优势不能自动变成当前split的已证优势。变更任务、schema、候选版本后，旧issue不能照搬。

决策门分开处理：

1. 动作合法且依据一致：执行。
2. 动作合法，但只有自由因果解释无法判定：仍可执行，解释保持unverified，不能写成已修复。
3. 明确的任务/配置/引用矛盾：暂停该提案，记录最小结构化问题，允许在剩余规划额度内做一次新的语义修复动作；仍由模型明确选择，不暗中换候选或改它的意图。
4. 网络/响应解析/served-model/用量/结算失败：停止付费队列，不进入语义修复，不重复原请求。

修复只适用于API已成功、JSON可读、usage已知且已结算的回复；任何ask异常都不能捕获后自动重发。修复请求计入同一路径最多3次调用，不能追加第4次。用完额度仍无合法提案，保留partial并报告未完成。修复措辞不等于关闭issue，必须让新提案通过对应机器检查；测量是否变好另记，不等同语义是否修正。

```python
def decide(state, call_budget):
    facts = registry_from_current_artifacts(state)
    issues = refresh_issues(state, facts)  # scoped; no causal truth from score alone
    response = ask_once(facts, issues)     # transport/accounting failure propagates
    proposal, checks = verify_structured_proposal(response, facts)
    if checks.legal_and_consistent:
        return proposal
    if checks.only_unverified_interpretation:
        return proposal.with_unverified_reason()
    save_issue(checks, raw_response=response)
    if checks.semantic_repairable and call_budget.remaining > 0:
        return request_bounded_repair(state, checks, call_budget)  # new metered semantic action
    return stop_with_verified_checkpoint()
```

B仍决定explore/confirm/stop及fit预留；A只处理下一提议的证据/契约与有界语义修复。不得把改动悄悄升级为新的B排序器。

## 4. 优先级P1条件扩展：按新证据重验证的短程计划

目标是降低重复委派次数，不是把系统退回固定的一次生成。每次最多规划未来两个explore候选，使用有限条件DSL；不输出任意Python、训练代码或额外fit。B提出explore需求时，A先验证计划片段是否仍适用。

PlanFragment含task/catalog/预算来源、候选ID、适用证据、有限if条件、预期失效条件。每一步执行后：用新真实反馈验证分支、候选可用性和预算；不成立或unknown时使计划失效，再在原3请求额度内重规划。confirm/stop仍B拥有，不能因为计划还有候选就强行探索。

```python
if scheduler_action != 'explore':
    follow_existing_scheduler()
elif fragment.valid_for(task, catalogue, remaining_budget) and branch_is_verified(facts):
    use_fragment_candidate_after_normal_validation()
else:
    fragment = ask_for_short_plan_within_original_request_cap()
```

风险：一次回答更长、计划可能过时、减少调用可能降低及时调整能力，尤其H1已说明LLM有时能探索到另一条有效路线。需同时测总输入/输出/推理tokens、调用数、失效重规划、实际方法效果，不承诺少两次请求就一定更便宜或更好。

## 5. 优先级P2条件扩展：带资源继承的安全续跑

A141现有恢复工具保留，不能改称已支持resume。新设计需增加事务式state提交、单调event序号、proposal/worker唯一ID、已消耗额度继承。只有完整提交的模型/预测/资源证明能被复用；损坏最新状态回退必须记录。

本地request_id不意味着学校服务支持幂等。未收到结果/usage的在途API不自动重发；原12历史授权不覆盖未来未知。worker若中断，保留attempt/completion差异；即使以已验证admission上界保守预留预算，也不能把这个上界报成实际完成fit数。续跑应优先处理本地安全边界，不重放付费未知请求。

这项主要提升可靠性与避免重复消耗，不能当分类提分。排在P0之后，不能挤占主要纠错证据与最后演讲时间。

## 6. 实验顺序、门槛与预算

### A145：事实/问题协议离线先验收

新48程序oracle案例，6类×8：对象身份、指标/分区/参照、参数方向、因果/切片作用域、失败/缺失证据、跨任务/版本失效。每类4合法/2矛盾/2未知，开发与保留各半、各含2/1/1；只使用当前真实注册的候选ID，避免旧未注册线性SVM夹具问题。先冻结输入/预期/源码哈希，再实现。

两版接收同一语义命题的旧表达和新结构化表达；结构化可核验覆盖率不等于理解任意英语。保留组只评一次，开发失败如实保留；同一开发者设计的程序oracle不称独立人工金标准。旧48/24、A120案例只作原样回归，不追其满分、不替换旧分母。

门槛：保留组错误确认0/6、未知误确认0/6；合法正确确认至少11/12，避免全unknown刷安全；默认旧行为/七类菜单/数据分区/3请求与11fit守卫不变。0fit/0API，预计实现与调试2–3小时。

### A146：少量真实请求，分别测自然决策和受控纠错

自然部分：Coupon/SMS/Flowers各一个预先固定完整历史前缀，当前compact与grounded+issue各一次，共6请求。不按响应结果挑例。比较合法动作、证据引用、明确矛盾、unknown/覆盖率、改动候选、tokens/费用/延迟；记录无自然错误时的天花板。

受控部分：3个固定真实历史状态各注入一个可由机器证明的错误记忆，当前反馈/新闭环各一次，共6请求。底层真实测量和候选不变，注入标签只留在离线oracle；报告纠正指定错误、错误记忆是否仍被引用、合法提案及误拒。故障注入不计自然错误率、不能冒充真实任务质量。

首批合计最多12请求、0fit、估计停止阈值¥.60，每回复一次无自动重生成。输出先封存后评价，只有契约合规度/可核对事实有机器真值；自由语义仍无人工金标准。原则：自然部分不得明显增加无效提案；受控错误的修复数增加且合法例不因所有拒绝而损失。样本小，只作继续门，不报显著性。机器约5–15分钟，含准备评价约1–2小时。

### A147：只有动作不同，才花训练预算验证它是否更好

对自然部分每个不同(candidate,params,原训练行,seed,预处理/资产版本)组合，在两个共同训练内评价分区执行。提议时两份评分不可见，外层/课程test不用于筛选。每个动作最多2fit，三状态×两组×两分区上限12fit；相同动作只算一次，已有精确同身份模型可复用。

只说措辞更正确、引用更多或换了候选，不算算法效果改善。报告配对胜/平/负、主指标差、失败、成本；若动作全相同，结论限定为可靠性/上下文结构，不为制造分类收益重训。预计5–30分钟机器，昂贵候选按已登记超时停止。

### A148：有前瞻信号后，才做完整Agent确认

先固定两个不同模态任务（优先类别表格和单文本）及训练内开发划分，当前compact与新闭环各完整自主运行，2×2路径，上限44fit/12Flash、每路径11fit含refit/3请求、单fit300/路径1200秒。统一同一生产快照的显式开关、候选、主指标和实际行号；新结构不能偷加方法或OOF训练。

仅当A145可靠性门通过、A146显示问题确实被纠正且不靠拒绝刷分，并且A147有动作质量信号或自然完成率改善，才启动。若只有受控可靠性改善、自然动作完全相同，可将其作为可靠性成果，跳过44fit矩阵。这里是已见任务开发证据，不能叫新unseen。

门槛预先采用各任务单独报告：完成率不低、无错误事实被确认；方法质量不得出现超过1个百分点的观测退化，至少一个任务有收益或完成率改善；如只有可靠性收益就不声称准确率贡献。1pp是开发决策容忍线，不是统计非劣证明。是否列为部署候选在开发阶段按登记规则决定，然后冻结，在另一个未用于调规则的真实任务评价迁移；迁移不能反过来重选F*，若据其改规则则降为开发证据。机器预计15–40分钟，路径硬上界80分钟加交付；4条付费路径估计阈值合计¥1.20。

### A149/A150：条件扩展，不一次全开

A149短程计划先做72个离线分支/失效/预算状态检查（不算真实任务），再选一个已登记开发任务做逐步请求/片段复用两组完整路径，最多22fit/6Flash/¥.60。要求真实调用至少少1次、总tokens/费用不增加、无明显质量/完成率退化，失败重规划仍≤3调用。通过后才考虑第二真实任务，不按Banknote是否满分写特殊规则。

A150续跑先复用已有模型做中断状态注入0fit/API，再做最多3fit的匿名真实worker中断探针。验证不重复已完成worker、额度不重置、在途API不重发、partial不冒充full。不要把已有A141九项再算新增独立证据。

## 7. 执行取舍与汇报方式

主线先A145→A146，首轮最多12请求/0训练；动作不同才A147（至多12fit）。A148是有信号后的44fit/12调用，不提前排满。主线全部条件满足上限56fit/24调用，不是已花费，也不是保证会用满；可选A149/A150另记，不和主线同时叠加。当前1636fit/375请求不变。

建议第一个6–8小时工作窗口完成P0实现、离线门和小型真实请求，保留全部失败；第二窗口只执行有信号的动作/完整路径。若研究时间不足，优先交付可核验决策+真实纠错机制证据，短程计划和续跑保留条件扩展；不压缩原计划最后两天演讲。

每小批开跑前：独立checkout固定提交、计划/runner、任务/状态/候选/行号SHA、fit/API停止额度。每次汇报用“瓶颈→结构→实验→证据边界”，分别给准确率/完成率/错误确认与覆盖、fit/调用/tokens/实际耗时/费用。不能用解析通过数替代真实性能，也不能把LLM候选变化说成提分。

结论允许是：更可核验但没提分；能纠正受控错误但自然收益未证明；更少调用但质量退化；或没有收益。未通过不晋级默认，不补请求或换种子追正结果。
