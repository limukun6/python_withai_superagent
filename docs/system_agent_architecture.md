# 从实验失败到系统架构：整个算法 Agent 的重设计

2026-11-17 起草；2026-11-17 已实现第一阶段运行时，但尚未运行新增的付费 Pilot 或正式比较。

## 实施状态（2026-11-17）

已落地并有本地端到端验收：统一 `run_task`、TaskContract、事实/假说校验、确定性聚合诊断、独立重载验收、F/S/C 数据隔离、冻结 top-2 的一次性 C 复核，以及 v2 `d/o/f/r/g` 对照策略。v2 的两个共同 warm-start、`d` 固定后续序列、`o` 单次规划、`f/r` 的反馈可见性差异和 `g` 的初始/失败/停滞触发均记录进运行日志。

尚未落地或不能据此声称完成：受限声明式 pipeline 组合、完整证据图、盲审 challenger、持久子组错误触发、组 bootstrap、正式 Pilot/E1/E2/E3、官方历史价格核验和未见任务迁移。因此本文中其余内容仍是设计与预登记草案，不能写成实验结果。

本文件是整体架构主方案。`evidence_driven_agent_design.md` 是其中“搜索、诊断和复核子系统”的详细设计，不代表整个 Agent。本文优先规定任务理解、推理约束、工具能力、运行时、记忆、交付和实验归因；子方案参数只在对应机制实验中使用。

## 1. 总体判断与研究定位

当前 Agent 的瓶颈不是单纯反馈不足，而是任务目标、算法知识、实验行动、统计证据、执行状态之间缺少明确契约。LLM 生成看似合理的解释后，系统往往只检查 candidate ID 能不能用，不能检查解释是否对应任务、是否有证据、行动是否能检验假说。

我们不能直接“修复模型内部思维”，也不能把一段理由当作内部认知的可靠观测。可以修复的是外部决策过程：使错误假设显式化，使后果受约束，使关键结论必须经过数据、程序或实验检验，使失败能够被区分、追溯和恢复。

目标架构是一个有确定性运行核心的算法研究 Agent。LLM 提议，程序检查可执行条件，工具产生证据，评价器按冻结规则选择。逻辑职责不等于必须为每层启动一个 LLM。

研究主张：相同模型与资源上限下，用任务契约、证据约束和执行验收提高有效决策率及任务完成率，并检验准确率与成本是否改善。不能用角色数量或对话长度证明自主能力。

## 2. 整条链路的瓶颈清单

状态说明：O=本地日志可观察；C=源码可确认；H=值得测试的假说/风险，尚非已证实原因。

| 阶段 | 瓶颈与证据 | 对应机制 | 不能保证的事情 |
|---|---|---|---|
| 系统集成 C | cli→runner/planner 与 benchmark→Policy/ExperimentLLM 两套核心；legacy遍历全部候选，选择顺序通常难以改变候选全集 | 一个run_task内核，策略由配置切换；同一客户端、账本与导出器 | 接口统一本身不提高准确率 |
| 任务理解 O/C | 文本对排序被解释为词汇相似度，swap被称作symmetry；loader只支持特定字段 | 带来源的TaskContract，声明标签和变换语义 | 类型检查不能证明所有自然语言解释正确 |
| 工具理解 O/C | 实际词bigram+LR，理由可称unigram等；只给参数与ID | PipelineSpec从真实注册表生成，允许动作有前置条件 | 元数据正确仍可能选择不佳 |
| 推理 O/H | “gap大→强正则化”反复出现，但缩小gap未提高准确率 | 事实/假说/预测分离，替代解释和受控对照 | 不能仅凭表现称模型具有某种心理偏差 |
| 探索 O/H | 表格feedback沿CatBoost局部调整；当前只有3个名额 | 维护不同表示/算法分支，显式选择探索或深化 | 更多探索不必然优于局部搜索 |
| 行动能力 C | 主要从14个固定配置选ID，无法主动要求诊断或组合新pipeline | 分层动作：inspect/compare/propose/train/repair/stop | 任意代码生成并非通用性的充分条件 |
| 评价 O | seed22表格内部更高的候选，官方验证反而更低 | 独立内部复核、配对预测比较、外部评价隔离 | 复核仍有噪声和选择偏差 |
| 资源 O/C | depth8到120秒超时；reasoning截断；未知费用后重试未扣风险预留 | 统一资源调度、预留、截止时间、错误分类 | 无可信token上界不能宣称绝对硬限额 |
| 审查 O/H | 18次review只改2次提案；相同信息下附和可能发生 | 审查任务是找可检验反例，必要时盲审；程序裁决 | 换角色/模型不等于独立或正确 |
| 记忆 C/H | 历史主要是线性记录，没有假说适用范围与撤销关系 | 结构化证据图与有作用域的假说状态 | 记忆更多也可能放大错误 |
| 恢复 C | planner失败与训练失败采用不同路径，fallback可让流程继续但策略已改变 | 明确状态机；语法修复/资源修复/策略退化分开 | API恢复不能证明算法更强 |
| 交付 C/H | benchmark选中模型与legacy predict接口并非统一交付协议 | 独立重载、行ID/标签检查、统一artifact manifest | 文件存在不等于可复现预测 |
| 结论 O | 报告误称超时全属feedback/reviewed，夸大审查无效及收益上限 | 数字由程序生成，文字结论引用证据及范围 | 自由文本的所有夸张仍需审阅 |

以上事实不否定已经完成的24组主实验。它们说明现有成功运行仅覆盖部分能力，不能自动推广成完整算法自主研发。

## 3. 总架构：一个内核、三类决策、四种验证

```text
任务说明 + schema + 允许使用的数据
           │
      TaskContract Builder
           │
  deterministic schema/semantic checks
           │
  Unified Controller ───── Budget / Event Store / Artifact Registry
           │
           ├─ Scientist: 提出相互可区分的假说与实验
           ├─ Pipeline Builder: 将方案编译成受控pipeline
           ├─ Diagnostic Tools: 提供新的可观测证据
           └─ Optional Challenger: 提出反例/区分实验
           │
     Type & precondition checks
           │
     Isolated Worker → Evidence Graph → 下一步行动
           │
  Freeze shortlist → Internal Confirmation → Freeze Model
           │
  Independent Acceptance Runner → Prediction Artifact
           │
  External Evaluator / Reproducible Report
```

三类决策：

1. 程序必须决定：预算是否足够、数据是否可读、候选是否合法、统计量是多少、失败类型、是否超过重试次数。
2. LLM适合提议：目标解释、候选方案、不同原因的假说、下一项有信息价值的实验、有限范围的修复。
3. 必须由新证据决定：哪种方案表现更好、假说是否得到支持、代码是否可执行、导出模型是否正确。

四种验证：结构/schema验证、行为/性质测试、统计实验验证、交付重载验证。不要把这四类都退化为“让另一个LLM看一眼”。

## 4. TaskContract：先保证在解正确的问题

字段：输入类型、样本单位、目标含义、标签映射、指标及方向、分组规则、允许增强、禁止读取的数据、输出格式、预算、声明来源与unknown字段。

例如，文本对可能是排序、相似性或蕴含。只有说明确认“标签表示A胜过B”，且平局等情况定义清楚，才允许swap(A,B)→1-y。相似度任务一般不能这样翻标签。

构建流程：LLM提取字段及来源片段；程序核对列名、类别数、缺失项和显式约束。关键语义未知时再次读取说明或执行不依赖该假设的通用基线，关闭相关增强；不得为保持全自动而猜标签含义。仍无法完成则输出具体不支持原因。

任务名只用于日志，不进入算法选择特判。任意新文件格式先由适配器转为统一schema；读取格式与选择算法是两个独立步骤。

性质测试示例：预测行顺序与ID可逆；标签编码/解码可逆；单行与批量预测一致；若contract明确要求交换反对称性，再测试该性质。不是所有任务或分类器都应强制swap概率严格互补；容差和适用条件要写在contract中。

## 5. 科学决策层：约束AI常见判断错误

### 5.1 三种陈述必须分开

- Observation：工具测得，引用run_id/row_scope/metric_id。
- Hypothesis：解释观察的候选原因，含替代解释和适用条件。
- Prediction：若假说成立，哪个受控干预应改变哪个指标。

例：训练0.98、搜索0.69是Observation；“正则化不足”“表示信息不足”“划分波动”是三个Hypothesis；“固定表示，仅调C后搜索准确率提高”是可检验Prediction。训练准确率下降不是目标成功。

每次最多两个主要竞争假说，避免不断生成文字耗尽预算。解释相同、无法区分的两个假说合并。没有可行区分实验时标记不确定，不装成已经知道原因。

### 5.2 假说是可撤销状态，不是聊天中的结论

保存字段：证据ID、父假说、干预、预期方向、控制变量、结果、混淆因素、作用域、状态。状态为未检验/初步支持/不支持/无法区分/执行失败。

一次搜索提升只能支持该次条件下的结果。不得将“C=0.3这次差”升级成“强正则化永远差”。API错误、OOM、timeout不能否证算法质量假说；它们只能更新可执行性或资源假说。

### 5.3 决策前的短协议

```text
我确定的事实有哪些？引用工具记录。
当前最重要的两个解释是什么？
哪个动作最能区分它们，或最可能直接改善任务目标？
该动作和当前方案究竟改变什么？
若失败，是假说失败还是执行失败？下一步如何区分？
```

这是一份结构化决策摘要，不要求保存模型隐藏推理，也不认为理由越长越可靠。

### 5.4 防锚定与盲审

保留至少一个不同表示/算法分支；当局部分支停滞且存在预算内的合理替代方向时，允许转向。探索比例是可消融的调度规则，不硬性规定每轮一定换族。

仅在关键分歧、未知任务语义或高代价动作前启用challenger。先让它根据contract与原始证据独立提出风险，再显示当前提案；其输出必须是“可核对的冲突、失败例子或区分实验”，不是投票同意/不同意。

盲审仍用LLM，错误可能相关；其观点不拥有否决实测结果的权力。无可检验的新内容就不启动额外训练，记录此次审查没有增量证据。所有额外调用计费。

## 6. 行动层：从选菜单扩展到可组合pipeline

行动分层：

- observe：请求已允许的数据画像或错误切片。
- compare：比较同一数据划分上的两个模型预测。
- propose_pipeline：组合注册组件，改变表示、预处理、分类器或参数。
- train：运行验证过的pipeline。
- repair：针对可定位的接口/资源失败生成有限修改。
- finalize/stop：冻结或保留已成功模型。

第一阶段用声明式pipeline配置，而非任意Python：

```json
{
  "input": "mixed_tabular",
  "numeric": {"imputer": "median", "scaler": "standard"},
  "categorical": {"encoder": "one_hot", "unknown": "ignore"},
  "estimator": {"name": "logistic_regression", "C": 1.0},
  "fit_scope": "fit_partition",
  "metric": "accuracy"
}
```

编译器验证输入类型、稀疏/稠密支持、标签类别、组件依赖和fit范围，转换为真正执行的pipeline。审计说明从该pipeline生成，减少“口头说LR，实际跑其他模型”的漂移。

数值超参数局部搜索可交给确定性搜索工具或经验证的优化器；LLM主要判断表示、模型族和诊断方向。优化器每次fit同样占预算，不能将多次内部训练算成一次Agent实验。

只有确有注册表表达不了的需求时，才考虑受限制的代码补丁。代码生成需要真实沙箱、离线依赖白名单、可读数据限制和独立测试；普通subprocess不等于安全沙箱。两人项目第一版不开放任意代码执行。

## 7. 证据图与运行记忆

Evidence Graph保存：contract版本、数据划分哈希、pipeline哈希、experiment、prediction、metric、hypothesis、decision、artifact及其引用关系。

原始工具结果只追加，不让LLM修改已记录的准确率。摘要可由程序重建。模型只收到当前相关证据及失败状态，不把完整历史无限堆入prompt；必要时可请求指定证据ID的细节。

所有证据含scope：当前任务/当前划分/当前pipeline/仅低保真。低保真结果不与完整训练结果直接排名。跨任务记忆默认关闭；若启用，只存可审计的通用经验并隔离评价结果，单独测试是否引入特定任务捷径。

记忆的作用是保留可以撤销的经验，不是永久相信第一次解释。

## 8. 运行时与恢复：资源也是任务的一部分

Controller统一管理状态：INGEST→CONTRACT→BASELINE→SEARCH→FREEZE→CONFIRM→ACCEPT→EXPORT。任何阶段失败都有明确出口；CONFIRM之后不能回到SEARCH。

失败分类：schema、pipeline编译、预测格式、训练资源、API传输、LLM格式、usage缺失、指标计算、导出校验。分别恢复，不统一“再试一次”。

例如：JSON格式错可以有界纠正；不支持稀疏输入应换组件或受控转换；OOM可以换资源更低的方案；用量不明暂停后续付费调用；训练超时保留已成功incumbent。恢复与算法改进分别计数。

每次行动先检查数据权限、名额、时间、费用预留；运行后按实测结算。没有可信成本模型时给范围与unknown，不能由LLM编一个“成功概率0.9”参与伪精确公式。

初版按规则排序：先排除不可执行动作，再保留合理资源余量，再比较候选的信息价值或目标收益。后续有足够真实记录才拟合成本模型，并报告预测误差。不得将训练平台环境差异解释成智能差异。

## 9. 最终评价与交付

沿用搜索子方案的F/S/C与官方validation隔离；最终指标由程序计算。LLM不能凭自信选择最终模型，不能自行宣布任务已完成。

Acceptance Runner在新进程重载模型和预处理器，在保留的内部样本上检查预测一致性、标签映射、行ID、空输入/未知类别行为和必要文件；正式测试输入只在冻结后按课程要求使用。验收通过才设置completed=true，strategy_valid与usage_complete是另两条独立状态。

报告数字自动来自账本。结论附适用任务、样本量、资源配置及证据路径；关于“所有超时都因X”“已接近最佳”等断言必须有覆盖范围和对照，否则降为假说。

## 10. 总控制伪代码

```python
def run_task(package, config):
    runtime = UnifiedRuntime(config)  # 同一核心用于CLI和benchmark
    contract = extract_contract_with_sources(package.instructions)
    contract = check_schema_and_explicit_constraints(contract, package.schema)
    if contract.has_critical_unknowns:
        contract = resolve_from_allowed_sources_or_disable_assumptions(contract)
    data = prepare_scoped_data_handles(package, contract)
    registry = compatible_pipeline_components(contract)
    runtime.record_contract(contract)

    incumbent = run_baselines(registry, data, runtime)
    while runtime.can_search():
        facts = runtime.evidence.visible_search_facts()
        issues = diagnose_task_data_model_execution(facts)
        proposal = scientist.propose(contract, facts, issues,
                                     alternatives_limit=2,
                                     budget=runtime.remaining())
        action = compile_typed_action(proposal, registry)
        checks = check_preconditions_provenance_budget(action, runtime)
        if not checks.pass_:
            runtime.record_rejection(action, checks)
            if not bounded_repair_possible(checks, runtime):
                break
            continue
        if high_value_review_trigger(action, facts, runtime):
            challenge = challenger.find_testable_counterexample(contract, facts, action)
            action = resolve_with_checks_or_budgeted_probe(action, challenge, runtime)
        result = runtime.execute(action, data.search_scope)
        runtime.append_immutable_evidence(result)
        update_scoped_hypotheses(action, result)
        incumbent = update_incumbent_by_frozen_rule(incumbent, result)

    shortlist = freeze_search(runtime)
    selected = confirmation_selector(shortlist, data.confirmation_scope)
    artifact = freeze_artifact(selected, contract, runtime.manifest)
    acceptance = independent_acceptance(artifact, data.acceptance_inputs)
    return export_with_status(artifact, acceptance, runtime.ledger)
```

## 11. 实验不能只看最终准确率

### A. 机制单元实验：AI错误是否被拦住

建立预先固定的行为测试集，每类包含合法案例与错误案例。开发集与最终检查集模板分开；正式检查前冻结代码。可以合成数据，合成事实只用于测试，不混入真实任务日志。

| 测试 | 注入/变换 | 成功标准 |
|---|---|---|
| 目标语义 | 同为文本对，分别是排序/相似度/蕴含 | 不把所有任务都套swap翻标签 |
| 任务名锚定 | 只改无关数据集名，语义和数据不变 | 不出现名称专用路由；统计比较输出与质量稳定性 |
| 工具幻想 | 不存在参数、错误估计器描述、多分类配二分类损失 | 执行前拦截；合法候选不过度拒绝 |
| 无证据断言 | 引用不存在run_id或篡改指标值 | 程序发现引用/数值不一致 |
| 轻信gap | 构造gap减小但search下降的真实结果表 | 不把gap改善当成目标改善 |
| 因果过度归因 | 同时改变表示与C | 标为多因素探索，不声称单因素因果 |
| 锚定提案 | 同一证据先展示正确/错误建议或不展示 | 比较错误采纳率；不要求LLM字符串逐字一致 |
| 候选顺序 | 打乱相同候选顺序 | 测量选择偏移和质量变化，不把随机性等同bug |
| 执行故障 | timeout、OOM、响应截断、usage缺失 | 进入正确恢复状态，遵守预算，保留incumbent |
| 交付伪成功 | 打乱预测行、丢编码器、标签映射错 | acceptance失败而不是completed=true |

指标包括错误拦截率、合法动作误拒率、恢复成功率、额外token、额外耗时。不能只统计拒绝数，否则“全部拒绝”也会看起来安全。LLM-as-judge可辅助归类文字，但数值/权限/格式事实以程序检查为准；语义样本由两名组员按预先标准复核。

对比：同模型直接提议、同模型加文字自检、提议加程序与证据校验。token上限相同并报告实际消耗。这样检验“架构约束”是否比“再想一遍”更有效。

### B. 系统实验：错误减少是否转化成任务收益

先统一核心和能力池，再比较：O一次规划、F反馈、S完整科学决策（任务契约+结构化假说+工具证据+统一执行）。各方案共享同样的可选pipeline组件和数据权限，保持资源上限一致。若S新增动作能力，O/F也能调用同样工具，否则只能称整体产品比较，不能隔离推理架构效果。

安全与数据隔离在所有臂保持开启。主指标为最终任务完成率、外部准确率、成本、时间；另报无效提案率、重复失败率、执行失败与统计失败的区分能力。沿用子方案预登记种子和配对规则；不要为了模块增多立刻再开完整巨大网格。

### C. 归因消融：分阶段而不是一次跑所有组合

先用A选出有可测收益的机制，再选择两项进行主消融：

- S去掉语义契约提取，仅保留同样底层数据安全：测语义错误与任务完成。
- S去掉竞争假说/控制变量约束：测有效实验比例与最终性能。
- S去掉独立复核：测固定轨迹上的选模损失，复用已冻结候选。
- S去掉结构化记忆，只给同预算的历史摘要：测重复错误与上下文成本。
- 同证据普通review / 盲审 / 可执行反例review：只在高价值节点比较，所有额外调用计账。

这些消融回答不同问题，不能仅凭累加模块后分数变高认定每个模块有效。保留负结果和合法动作误拒案例。

### D. 泛化验收

在未参与提示词、规则和适配开发的新任务上运行冻结系统；包含至少一种新的标签结构或schema。只改数据集名字不是泛化实验，多seed也不等于多任务。若某模态尚不支持，报告该边界，不用固定任务专用实现冒充通用处理。

## 12. 两人项目的最小实现与优先级

P0：统一内核/客户端/账本/导出路径；现有benchmark策略接入同一run_task。数据边界和预算守卫在所有策略下相同。

P1：TaskContract + 真实PipelineSpec + 结构化事实/假说 + 类型检查 + 独立交付验收。先解决做错任务、说错工具、运行假成功，做A类低成本测试。

P2：诊断工具、竞争假说、局部分支、C复核。采用子方案E1/E2中的配对设计验证。

P3：按需challenger、gate、成本预测、声明式pipeline扩展。仅实现已有失败证据支持且单元实验有效的机制。

P4：新增模态适配与受限代码生成。后者不是前几阶段完成的先决条件，也不适合第一天就开放。

人员A负责运行内核、数据权限、pipeline编译和验收；B负责contract/假说协议、诊断、测试集与统计。双方交叉审核语义案例，保证都能解释提交系统。

不把所有模块实现成独立AI：常驻LLM逻辑角色仅Scientist；contract提取按需调用，Pipeline Builder优先程序实现，challenger按触发调用，evaluator/资源控制/验收用确定性程序。

## 13. 研究依据与限制

- [Large Language Models Cannot Self-Correct Reasoning Yet](https://arxiv.org/abs/2310.01798)：研究发现，在其测试的推理场景中，缺少外部反馈的自我纠错可能失败甚至降低表现。不能据此断言今天所有模型、所有任务都不能自纠错；它支持在本项目中检验外部证据的必要性。
- [Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657)：将观测的多Agent失败归为系统设计、Agent间失配和任务验证等类别。这支持把诊断范围扩大到整条执行链，而非只调一个planner prompt；其基准不是本项目，效果仍需本地验证。
- [AIDE](https://arxiv.org/abs/2502.13138)、[Reflexion](https://arxiv.org/abs/2303.11366)：可借鉴搜索组织和经验记录。本文组合是课程工程研究设计，不声称新发明通用认知架构。

最终成功标准：系统能指出自己知道什么、不知道什么、下一步为什么值得做；错误有可观察的拦截或恢复；任务输出经独立验收；复杂度增加确有实验依据。模型仍可能提出错误假说，但一次错误不应未经检查就贯穿整个系统。
