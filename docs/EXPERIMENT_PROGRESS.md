# 实验持续工作记录

## 2026-11-24：A149 片段探测，模型两次都没有返回 `plan`

按预登记执行单次探测（sms，fit 上限 4、调用上限 3、`--plan-fragment advisory`）：**模型两次提案都没有返回 `plan` 字段**，`plan_fragment_offered` 与 `plan_fragment_rejected` 均为 0。已核对片段指令确实进了 payload。

按预登记"模型从不返回 plan 即停"，不再调提示词重试。结论：片段机制代码可用（零成本四组已证），但模型没有动机产出；`--plan-fragment` 保持默认关闭。

attempt_01 无效：运行器漏传 `--plan-fragment`（我的字符串替换 bug），该次 ¥0.022225 只算普通运行。有效部分 ¥0.021502。合计 4 调用 / 8 fit / ¥0.043727，无新增 unknown。账本 457 请求 / known ¥7.925792 / unknown 16。报告 batches/A149_fragment_probe_results_20261124_zh.md。

## 2026-11-24：A149 片段接入执行入口，零成本验证通过

`--plan-fragment advisory` 已接入决策循环：片段步骤成立且候选可选时直接采用、不调用模型；调度给 stop/confirm 时一律作废。

新增 `deferred` 状态：只有 `reference_trial` 未满足时继续等待，其余条件不成立才作废——实测发现原实现会把"还没轮到"误判为"永久失效"。

零成本四组对照（本地决策、0 API）：有效片段把 trial 2 从 recipe 候选改成 `single_word_char_C3.0`（0.920173 → 0.947372）；永远等待、永远不成立两组都退回基准且结果逐位相同。三种状态在真实运行中均出现。

不声称提分（单次单任务）、不声称省调用（本轮无请求）、不声称片段可靠（片段是手工构造的，模型产出合法率未测）。测试 143 项全绿。报告 batches/A149_fragment_wiring_20261124_zh.md。

## 2026-11-24：A150 续跑执行接通，零成本端到端验证通过

`--resume-from` 已落地：新目录 + `adopt()` 抄证据 + 预算与执行器计数从父运行继承 + 五条硬约束（新目录、同 fit 上限与 seed、源码一致、任务包一致、账本已结清）。

零成本验证（本地决策，0 API）：一次跑完 3 fit、验证 0.920173080213678；人为中断到 1 fit 后续跑，得到**同一候选、同一验证分**，fit 计账 3 = 1 继承 + 2 新增。**中断 1 fit + 续跑 2 fit = 一次跑完的 3 fit，无重复训练。**

两个真实拦截：改过 `resume.py` 后源码漂移被拒；预算与执行器派发计数不一致被拒（修法是用父运行真实 fit 记录预置，而非伪造占位）。

未做：带 LLM 的续跑验证、显式幂等去重键。测试 142 项全绿。报告 batches/A150_resume_execution_20261124_zh.md。

## 2026-11-24：A150 只读续跑判定落地，续跑执行未接入

新增 `superagent/resume.py` 与 `python -m superagent resume-status`：给出 `complete / resumable / blocked` 三种判定，护栏覆盖账本未结清、账本落后、源码漂移、任务包变化、试验编号不连续、fit 未完成、预算不足、清单缺失。`inspect` 只读——测试断言账本哈希不变、运行目录不新增文件。

真实中断件验证（0 花费）：已完成的 `pair_s707_b2` → `complete`；被墙钟停止的 `text_s911_full` → **`resumable`**（0 fit / 0 请求，剩 3 fit / 3 请求，scope 账本一致）；不存在路径 → `blocked`（退出码 2）。

**未做**：续跑执行本身、预算继承接线、幂等去重键、以及"续跑后总账等于一次跑完"的验证——因为续跑动作还不存在。报告 batches/A150_resume_status_20261124_zh.md。

续跑锚点另有结论：**从日志重建状态做不到**（`llm_usage.jsonl` 无试验编号），但 **139 个 `state_k.json` 全部是其运行的忠实前缀快照**（history/prediction_evidence/预算计数逐项对账 139/139），所以续跑应锚定最后一个状态文件。证据 `research/a150_resume/state_prefix_check.json`。

## 2026-11-24：探索广度对照，加宽无分辨收益

8 条路径（2 任务 × 2 seed × 2 臂）：32 fit / 16 调用 / ¥0.203370，无新增 unknown。breadth2 = anchor + 1 次探索 + refit；breadth4 = anchor + 3 次探索 + refit。

配对差：成对文本 +0.851 / +0.851pp，单文本 −0.785 / +0.785pp，均值 **+0.425pp**，无一超过 1pp。按预登记规则（≥3 个正向且均值 ≥0.5pp），"加宽有收益"**不成立**；四个差全部落在 1pp 漂移带内，等价于"测不出"。

池最优上界（+1.81 / +2.12pp）没有被单次加宽兑现——"上界偏高"与"没试到池最优候选"两种解释本轮分不开。

代码：请求上限参数化为 `--llm-request-limit`（默认 3，允许 1..12），两处校验上限放宽到 12，5 项测试锁定默认不变。账本 437→453 请求、known ¥7.678695→¥7.882065、unknown 16 未增。报告 batches/A151_breadth_results_20261124_zh.md。

## 2026-11-24：A151 表示配对，compact 非无损且漂移达 7.9pp

16 条完成路径（含 4 条重复）+ 1 条被墙钟停止：48 fit / 16 调用 / ¥0.177365，无新增 unknown。

只换 `--menu-view` 的 6 对：4 对同选；单文本 707 +7.89pp、成对文本 707 −2.55pp。重复格显示单文本 707 的 `full` 视图两次给出不同候选（0.8684 vs 0.9474，**同配置漂移 7.9pp**），故 +7.89pp 是漂移；成对文本 707 的差异两次稳定复现，−2.55pp 是真实表示效应。漂移校正后 6 对均值 **−0.56pp**，两个可复现差异都指向 `full` 更好。

输入 token 降 25.6%（29,732→22,124），耗时反而 +62%（样本少、方差大），不能宣称省时间。默认保持 `full`；`compact` 不宣称无损，今后涉及它的对照必须重复。

另记：一条路径因机器休眠导致 `elapsed_seconds=859 > 600` 被共享墙钟停止（单调时钟含休眠时间），已在新进程重跑完成。

账本 421→437 请求、known ¥7.50133→¥7.678695、unknown 16 未增、pending 0。报告 batches/A151_menuview_results_20261124_zh.md。

## 2026-11-24：A151 多 seed 配对，grounded 6 个配对输 5 个

按预登记执行 2 任务 × 3 seed × 2 臂 = 12 条路径：36 fit / 15 次调用 / ¥0.238386 / 25 分钟，无新增 unknown。

成对文本 mean **−0.89pp**（−2.02 / −2.55 / +1.91，1 胜 2 负）；单文本 mean **−3.21pp**（−7.11 / −1.27 / −1.27，0 胜 3 负）。两任务均值均 ≤0、无任务 2/3 为正 → 预登记判定 `grounded_not_better=true`、`positive_signal=false`。

6 个配对两臂从未选中同一候选，差异全部来自选型；grounded 共 9 次调用 vs 普通 6 次。不是普遍规律（2 任务、3 seed 共用同一验证集），不做显著性声明。grounded 仍为可选研究开关，不晋级默认（本轮未改默认）。

账本 406→421 请求、known ¥7.262944→¥7.50133、unknown 16 未增、pending 0。报告 batches/A151_multiseed_results_20261124_zh.md。

## 2026-11-24：A151 矩阵五条路径全通过，17 fit／6 调用／¥0.067218

按预登记执行五行：三条迁移（类别表格、数值表格、单文本，calibrated-v2 + compact）、单文本 grounded 修复、一次真含 confirm 的复核验收。全部通过，每条 `requested_planner_completed=true`、fit 计数完整、无新增 unknown。

W1：压缩表示在三种新类型跑通；grounded 在单文本第 1 次被拒后修复接受（7 条断言全 confirmed）。W2：`ab-finalists` 动作 explore×2→confirm×2→stop，`basis=["primary","confirmation_1"]`，两 finalist 各在两划分测量（word_char 0.6547 vs word 0.6368），`independent_confirmation_observed=true`，为 A151 首次真正训练独立确认分区。

过程偏差留档：attempt_01 漏参 0 花费；attempt_02 因自建包无 test split 改用"重载+验证集"交付渠道；attempt_03 的 w2_confirm 被验收逻辑误判，离线重算 recheck.json 后五条全 pass，未重跑。

不声称提分/迁移收益/通用性/压缩省时：grounded 单文本 0.9347 低于 L0 的 0.9474，单样本且同向。账本 406 请求 / known ¥7.262944 / unknown 16 未增 / pending 0。报告 batches/A151_matrix_results_20261124_zh.md。

## 2026-11-24：A151汇总证据复核与口径修正，0新增实验

核对completion、actions、confirmation及delivery_check：两条路径均完成，grounded真实语义拒绝/修复/接受成立，两次独立进程预测SHA相同；但实际动作explore→explore→stop，无独立confirmation训练。修正汇总中L1费用4倍为2.4094倍（+140.94%）；成功三阶段合计6fit/4API/¥.065158，原6fit/3API/¥.055246只指端到端阶段。3/3知识纠错免fit仍是离线机制证据，不等于实测节省3fit。未新增训练或API，A1657/400请求/known7.195726/unknown16/pending0不变；不因一次可用而晋级默认。

## 2026-11-24：A151 汇总报告，三段结论与不可声称边界

汇总 A151 三段（分层机制 0 成本、单调用探测 1 调用、端到端验收 6 fit/3 调用）：成功阶段共6 fit、4次调用、28,506输入/9,163输出token、¥0.065158。其中端到端阶段为6fit/3调用、23,782输入/7,866输出token、¥0.055246、245.3秒，账本400请求/known¥7.195726/unknown16未增。早期失败、用户诊断和其它本地验收不含在此成功阶段小计内。

四类证据分别停在机制层（3/3 知识纠错免 fit）、表示层（字节 −35.039%）、真模型真请求（1 次被拒 + 1 次修复接受）、工程闭环（两条路径均完成且各 3,232 行预测）。质量结论不成立：L1 第二次探索 0.6216 低于 L0 同一步 0.6520，锚点保持不变。

报告 batches/A151_consolidated_report_20261124_zh.md。下一步决策点：grounded 是否降为受限实验开关、课程对运行时模型的要求核对；A149/A150 不在本轮。

## 2026-11-24：A151 端到端验收完成，6 fit／3 调用／¥0.055246

L0（calibrated-v2，分层 off）2 次搜索、1 次 Flash、3 fit，选中 `word_char_C0.05`（内部 0.6520 / 验证 0.7011）。L1（grounded-v1 + by-class）2 次搜索、**2 次 Flash**（第 1 次断言未知被拒，修复后 8 条断言全 confirmed 而接受）、3 fit，停在 anchor `word_C0.05`（验证 0.6755）。两条均 `requested_planner_completed=true`、无重试、独立进程各 3,232 行预测、源码前后一致。

账本 397→400 请求、known ¥7.14048→¥7.195726、unknown 16 未增、pending 0。**压缩表示下 grounded 链路真实跑通**。

不声称提分/省时/通用性/L1 优于 L0；L1 第二次探索取到 0.6216，低于 L0 同一步的 0.6520，是 A147 同向的负结果。报告 batches/A151_e2e_acceptance_results_20261124_zh.md。

## 2026-11-24：A151 单调用菜单探测成功，1 API／0 fit

按预登记执行 1 次真实 Flash、0 fit：`accepted=true`，选中 `word_char_C0.1`（原 29 菜单内、参数与目录一致），`calibrated-v2` 11 字段齐全、`finish=stop`，4,724/1,297 token、¥0.009912、52.7 秒、444 个 SSE 数据事件，**无新增 unknown**。账本 397 请求 / known ¥7.14048 / unknown 16 / pending 0。

对照原失败 6,821 输入、417 秒零事件、无 usage，本次输入 token 少约 30.7%。**仅一次提案**：无执行/反馈/复核/重训/交付，不是闭环；不声称压缩降低推理时间；不声称 grounded 可用（本次未开）。首次尝试因运行目录缺失在发请求前失败（0 请求），原件保留。

报告 batches/A151_menu_probe_results_20261124_zh.md；下一步为两条路径的端到端验收（每路径 ≤3 调用 / ≤3 fit）。

## 2026-11-24：A151 菜单表示接入生产开关，0 fit／0 API

按 Codex 复核第 4 节：新增 `superagent/menu_view.py`（`menu-tables-v3`）与 `--menu-view full|compact`（默认 full），只改表示。真实失败 payload 实测 21,473→13,949 字节（−35.039%），`decode_menu` 往返相等，29 个候选与参数逐项保留；提案仍按原 capabilities 校验。11 项 menu 测试 + 相关 9 模块 71 项回归通过。

单调用机制计划已落盘（batches/A151_menu_probe_plan），runner 默认只校验并已跑通；上限 1 次 Flash/0 fit，未发起真实调用。按复核撤回三处过头结论：客户端无默认重试、`other_events=0` 不能证明无字节、失败 payload 是 calibrated-v2 的 11 字段且无 grounding 包。

账本不变：396 请求 / known ¥7.130568 / unknown 16 / pending 0 / 快照 v3。A 累计 1651 fit。

## 2026-11-24：A151堵点二免费复核与菜单编码原型，0fit/0API

核实当前requests/urllib3默认零重试及连接/读取/协议错误分支；修正“417秒即七次POST”与“原16384预算必然推理耗尽”的过度归因，指出原失败在calibrated L0、无grounding。空行漏统计和iter_lines缓冲使零事件不足以证明零字节。9定向测试通过。原型1增2.356%字节、原型2少6.625%、共享列名+重复值原型少35.502%(20030→12919紧凑JSON字节)，全部29候选及原核验字段可还原，非token或真实模型效果。首次日志kind键误写导致免费拒绝保留原件，后续各原型独立目录。

生产与账本未改，无新付费/训练，A1651。按已导入8条用户诊断后的最新真账记录为396/known7.130568/unknown16/pending0(v3接受清单)，下文388/13过期。下一步代码开关/真实最多单调用探测未启动，不重放旧失败。计划/结论见batches/A151_blocker2_offline_review_plan_20261124_zh.md与A151_blocker2_codex_resolution_20261124_zh.md；runner scripts.review_a151_blocker2，证据research/a151_blocker2_codex_review_v4。

## 2026-11-24：A151 条件小试，1 fit／1 次失败调用，账本新增第 13 条未知

按计划执行（`ab-v1`、fit-limit 3、每路径最多 3 次 Flash）。L0 在第一次 Flash 提案处报 `ConnectionError: RemoteDisconnected`，按停止规则中止：1 次 fit、1 次失败调用、无自动重试、L1 未启动。部分交付保留 trial_1 检查点（6660/8301 行），独立进程 predict 输出 3232 行。

账本 387→388 请求，known 仍 ¥7.0165495（未增），unknown 12→13（新 ID `153d96d3…`，scope `a151_loop_off`），pending 0，SHA `5b1f5c58…`→`d671c424…`。新 unknown 不在已接受快照内，`check_snapshot` 会拒绝后续付费；处理前不再发起真实请求。A 累计 1651。

闭环未跑通（断在第一次提案），L1 未运行，无分数结论。另记：失败调用约 7 分钟才报断连，而文档读超时 60 秒，再次付费前需确认底层传输重试。两次尝试原件均保留：`runs/a151_small_trial`（0 花费的参数冲突）、`runs/a151_small_trial_attempt_02`（真实失败）。报告 batches/A151_small_trial_results_20261124_zh.md。

## 2026-11-24：A151 知识纠正与执行修复分层，0 fit／0 API

按交接第8节第一项实现`--issue-layering off|by-class`（默认off）。issue按命题种类分knowledge/execution/measurement；知识类由已核复述关闭为`resolved_semantic`并标`new_fit_required=false`，执行类等匹配执行证据，测量类不被无关成功试验关闭。默认strict/full与grounded-v1旧行为逐字节未改，旧6项测试未改全过，新增12项分层测试。

零成本验收：程序oracle 5/5；封存A146六条grounded响应回放，3条受控纠错全部由`addressed_pending_check`变为`resolved_semantic`（coupon/sms/flowers），6/6候选选择不变，0条知识类仍open。短信`inverse_C_between_bound_configurations`即A147起因，现免fit关闭。如实记录后果：诊断动作未被禁，但知识类关闭后该轨迹不再有可引用的open issue，故诊断提案会被契约拒绝；已写成测试。

本批0 fit/0学校API/¥0、0.985秒；15个相关模块94项回归通过。A累计仍1644fit；唯一ledger未改仍387/known¥7.0165495/unknown12/pending0、SHA5b1f5c58…，A空闲。报告batches/A151_issue_layering_results_20261124_zh.md，计划plan、证据research/a151_issue_layering。不宣称外层提分，A147负结果与A148决定不变。

## 2026-11-24：DeepSeek接续交接，0新增fit/API

用户额度将尽，整理docs/DEEPSEEK_A_LINE_HANDOFF_20261124_ZH.md作为单一接续入口：A/B职责、最新生产与结果版本、八类瓶颈/结构、完整数据流、各批资源与结论边界、唯一账本和已授权历史未知、演讲/流程图源、建议五小时工作与runner。校验全部97生产源、演讲源/交付SHA及真实ledger SHA均匹配。下一步知识纠正/执行修复分层尚未实施，A148不启动；交接不是追加实验授权或最终冻结。A1644/387请求/known7.0165495/unknown12/pending0不变。

## 2026-11-24：按反馈补清结构、全项目数据流与处理流程，0新增实验

用户指出初版结构不清。已核对ab_run/ab_scheduler/ab_flash/contracts/worker/导出/恢复代码，原位修订同一Spaces文稿：5页主讲35/35/30/25/25秒，5页隐藏附页；前3页为全项目数据/决策/证据回路、六项瓶颈与结构行为、处理循环及纠错/失败分支。附页9数据边界、附页10真实模块/对象；保留旧问答/来源和原生token图表。区分本地原始分区数据与Flash摘要，B调度动作与探索选候选，默认准入与可选grounded/compact，恢复交付与未实现续训。

结构文件presentation/ARCHITECTURE_AND_BOTTLENECKS_ZH.txt列8项瓶颈，含预算/稳定复核两项全程能力；两份.mmd留可编辑图源，revise_architecture.mjs留修订代码。明确AB状态为schema字典，不能硬说所有入口都实例化通用benchmark的DecisionState。新版约609汉字，备注及脚本同步，需按实际语速试讲。不同批次版本和当前97源真实完整grounded路径未验边界保留，不新开A148。项目0新增fit/API，仍A1644/ledger387/known7.0165495/unknown12/pending0，A空闲。

## 2026-11-24：A线2分30秒演讲材料已制作，0新fit／0学校API

按用户提供的口头4/视觉3/系统实验理解5评分，制作5页主讲+2页隐藏问答的个人Spaces演示文稿，逐页中文讲稿15/40/35/30/30秒共150秒，约554汉字另有少量英文。主讲覆盖任务/规划/证据/执行架构、A145/A146受控核验、A140输入token、A141恢复交付、A147负结果及成本取舍；明确不同批次版本和非自然错误率/非完整新源交付边界。7页全渲染检查，原生表格/图表保留编辑能力，备注含讲稿/来源/问答。

文件presentation/A_LINE_2M30_SCRIPT_ZH.txt、build_a_line_deck.mjs、delivery_manifest.json、README.txt；云Page ID见manifest。可在文稿查看器导出PPTX；主讲实际语速需朗读校准。没有启动新训练/API队列，也未调用学校模型；A1644/ledger387/known7.0165495/unknown12/pending0不变，A仍持有空闲。不把制作演讲材料当成共同最终冻结。

## 2026-11-24：A145–A147完成，8新增fit／12真实Flash，保留可选而不扩大当前variant

按已保存计划先冻结48程序oracle再实现97生产源473ba84：运行/任务/目录/生产者绑定事实、机器断言、带状态的纠错、可核验执行理由；显式grounded-v1，默认strict/full及B目录/调度不改。55回归/7菜单一致，开发24/24与一次保留24/24；错误确认0/6、未知误确认0/6、合法12/12。首轮43测试1夹具错误保留，修正后55通过；训练分数不借验证集SHA认证。

A146三真实已见任务同state_4×自然/受控×旧compact/新协议12请求全合法；新23断言通过，3受控问题明确纠正pending，0真实语义重发。旧组也能发现部分错误，未要求其输出新字段；原prose两标记为疑似其它分类器引用误报，不能当2条真实错误收益。新旧输入47106/27213、输出13899/8032、费用.102702/.059341，新的更贵约73.1%；全12请求74319输入/21931输出、课程折算.162043、726.625秒。

A147只追两个不同动作，primary707/confirmation_1 808同payload，8实际fit/0API/92.547秒。SMS受控新diagnose相对旧improve的macroF1平均低3.1840pp；Flowers自然coverage平均+.625pp，两个划分方向相反。四个同动作配对不重复训练，不把跳过量换算为实测省费。没有外层评价/最终refit，不能作完整Agent或新任务泛化收益。

当前不启动A148 44fit/12调用，不改默认；决定非预登记显著性检验。后续考虑知识纠正与执行/测量修复分层，尚未实现。A149/150未启动；新97源完整grounded路径和交付尚待未来验证。研发未最终冻结，H2/Released2仍B，旧模型源码不重标。

A累计1644；唯一实账本runs/a129_B_ledger_return/private/ledger.sqlite已387/known7.0165495/unknown12/pending0，原375 requests/transitions未改，A空闲；owner research/a145_a146_grounded_pilot/paid/current_ledger_owner.json。原12未知按用户授权保留null不再等官网，新未知仍停。报告batches/A145_A147_grounded_protocol_results_20261124_zh.md，证据research/a145_a146_grounded_pilot、research/a147_grounded_action_followup，失败原件/计划/复跑脚本已落盘。下文设计时1636/375为历史快照。

## 2026-11-24：A145–A150下一轮A线方案已设计，尚未运行

用户要求按当前瓶颈设计提升方向。基于fb791d5代码/既有结果，主攻可核验决策协议和有关闭条件的问题闭环；带失效检查的短程计划、资源继承续跑为条件扩展。复用现有measurement/configuration/metric/scope模块，不把已有基础或B调度称新贡献。A145离线48程序oracle，A146最多12真实请求区分自然/受控，动作不同才A147最多12fit，有信号再A148两任务四路径44fit/12请求；全主线条件上限56fit/24请求，不是已消耗。A149/A150另记，不同时全开。计划docs/A_LINE_NEXT_ARCHITECTURE_AND_EXPERIMENTS_20261124_zh.md及预算JSON已保存；本次0新增fit/API，A1636/ledger375/known6.8545065/unknown12/pending0不变，未启动队列。

## 2026-11-24：A142–A144新真实Banknote补充迁移完成

先预登记同94源/默认full7目录，官方一次公开下载，1372×4、1092训练/280封存，24重复特征同组无跨界，outer71424/内部707808909固定并签名。L0与EC0两完整路径各11fit，全封存后统一评分均280/280、F1=1、逐行相同；自主选ExtraTrees leaf1/leaf3，native/搬迁/隔离包一致，训练多数类参照56.79%。不是新图像/H2或最终Unseen，不宣称Flash增益、总体100%或单独压缩收益，不为追差异换seed。

新22fit/3Flash，输入15611输出4250、¥.032611；循环15.562/155.312秒（后者含API140.766），队列173.297秒，评估18.828秒0fit/API。A累计1636；ledger375/known6.8545065/unknown12/pending0，原372行及全部旧transition保持，A持有空闲。报告batches/A142_A144_new_numeric_transfer_results_20261124_zh.md、证据research/a142_a144_banknote_transfer，B继续H2/Released2，无活动队列。

## 2026-11-24：A141中断成果恢复9项通过，0新fit/API

新增独立tools/recover_run_checkpoint.py，核原训练source/包/指标/分区/prediction/fit，另目录恢复partial checkpoint或保留已有完整模型重交付。A138已有组件生成9工程快照，6恢复路径32行native/隔离一致、3错误拒绝；包含state截断回退、在途unknown保留、重复恢复原件不改。不是实际断电试验，不支持继续原训练/搜索/API。94生产源及实际372ledger均未变。累计1614fit/372请求，known6.8218955/unknown12/pending0，A持有空闲。报告batches/A141_interruption_recovery_results_20261124_zh.md，证据research/a141_interruption_recovery；目标/反馈/中断三项独立补证本轮完成。

## 2026-11-24：A139–A140紧凑反馈与9次真实Flash完成

第一版仅引用少3.268%附加字节、未过20%门，0付费负结果保留f61cfc3；另登记无损表格化，40检查/三上下文还原通过，附加字节减23.9898%/整请求约5.75%，同3已见工程状态不算独立验证。显式compact只checked，默认full不改。真实3任务×raw/full/compact轮换共9请求，9动作合法，三任务各候选均相同；输入10157/13157/12348，compact少809/6.149%输入token；总费.019945/.026421/.022268，compact比full少.004153含输出长度差，不宣称稳定15.7%或质量/纠错改善，仍比raw贵。自动prose矛盾0但unknown6/3/7，非人工真值。

9成功Flash、0fit，35662输入8243输出，¥.068634，238.359秒。A1614，实际ledger372/known6.8218955/unknown12/pending0，原363及旧transitions未变，A持有空闲。新owner research/a140_flash_projection/paid/current_ledger_owner.json；旧账本副本不用。报告batches/A139_A140_compact_feedback_results_20261124_zh.md、证据a139_feedback_v1/a139B_feedback_v2/a140_flash_projection。接续中断恢复零训练，B仍H2/Task2。

## 2026-11-24：A138目标稳健性完成，累计1614

先冻结18案例和四条3fit计划。正式CLI预训练截停18/18通过；四完整匿名路径在共同注册行号上检查accuracy/标签反向改名/binaryF1正类0/AUC正类0，每条自主线性→树→refit，各3fit，整体54.454秒，四独立包预测一致、标签映回32行一致、正类和概率列序区分正确。0API，无生产修复；同数据变换不是新真实任务或新质量增益。新增12fit/累计1614，ledger363未调用；接续A139紧凑反馈。证据research/a138_goal_robustness、报告batches/A138_goal_robustness_results_20261124_zh.md。

## 2026-11-24：A137可选数值核正式接入与匿名自主交付

计划/runner先行、隔离适配B5bd6b14旧补丁，数值核默认off、只按数值schema及资源资格进入共同full，原守卫保留、弱标签不接。七类默认菜单完全一致，46无fit/无网络检查通过；匿名240×3/40查询由正式local ab-v1/full/707/fit3自主试线性和核、选择核并refit，实际3fit/CLI7.437秒，native/搬迁/93源隔离包预测逐字节一致，不评分、不读课程/H。新增3fit/0API，A1602，账本363未改。它验证工程闭环，不把合成当真实泛化；真实核效果仍B Bean+16、Wisconsin未提分。Task1候选保持6b6ace8的92源不重标/不重跑；最终项目/test尚未冻结，H2/Released2仍B。报告batches/A137_numeric_adapter_results_20261124_zh.md，证据research/a137_numeric_acceptance，无活动队列。

## 2026-11-24：A136 B最新七项研究复算通过

B固定5bd6b14，SHA全清单、源码包/runner、七研究回执、实际fit、配对组区间及导出SHA全部通过；本批0fit/API，A仍1599，ledger363字节未改。Bean新数值核1989→2005/2178、+16/+0.735pp；动态7/固定11fit同预测。第二真实二分类任务核候选被试后拒绝，仍111/114/AUC.987553、多1fit，支持可用/可拒绝而非迁移提分。两Bean池重叠402组不算两任务；Task2传递弱标签+3未达+8门槛，不接。B恢复后39fit及更早4fit只引用历史，不加入A。旧c8补丁四文件上下文不匹配，只check未apply，生产92源不变。报告batches/A136_numeric_and_weak_label_review_results_20261124_zh.md、证据research/a136_numeric_review。下一批可单独适配默认关闭的数值核，零重复Bean矩阵；H2/Released2仍B。

## 2026-11-24：A133–A135可选推理路由、当前源Task1交付完成

计划先落盘再执行。B924e88a可选置信路由已接，默认off；两原始凭据验证器通过，初40/补CLI守卫后41项no-fit/no-network检查通过，不重复B性能训练。四历史推理相对完整组合正确数0/-2/-5/0，本地编码输入量省略不当作API费；保持可选。A134已有A127组件手工工程夹具8行、阈值0/1/.2，第二模型实际0/8、8/8、8/8，三种交付一致，29.328秒/0fit，不冒充自主性能。

A135当前92源、原训练SHA对齐、8117行/15候选、707/local ab-fixed/full/standard，4探索6复核1refit＝11fit，自主选字段交互树。训练88.578秒、整批101.093秒，封存后课程验证1523/2030＝75.0246%，三种2030行预测逐字节一致，原test未读；与旧结果相当，非新结构提分。首prepare属性错误0fit/API保留，v2完成未重复训练。累计1599，新API0/费0；实际ledger363/known6.7532615/unknown12/pending0未改，用户继续授权有效。

证据research/a133_a135_summary，报告batches/A133_A135_routing_and_task1_results_20261123_zh.md。H1已完成，H2/Task2仍B负责，最终两任务版本/test未冻结；九天里程碑顶部已纠正旧“缺48/24/C1/H未跑”。B5bd6b14只登记，未接数值核/弱标签；无活动队列。

## 2026-11-23：A130–A132按用户授权恢复付费，H1完整完成

用户允许原12未知不等官网账单继续，按ID/原行SHA显式接纳，不结算零、不删记录，清单外新增unknown/pending仍停止。外部准入wrapper保留91个分类生产文件与原L0模型一致，19项policy检查通过，首次大写预算ID0fit/0请求失败已修h1_transfer并保留。A131两R0各11fit/3Flash，22新fit/6成功计量，输入15890/输出6481tokens，¥.041814增量，队列352.719秒。A累计1588，ledger363/known6.7532615/unknown12/pending0，原357全部记录及转移未变。

四模型全封存后首次H1统一评分：61411 55.36→58.35%(+12/+2.99pp)，61421 55.25→60.00%(+19/+4.75pp)，四模型搬迁/隔离预测一致。R0在同8合法候选/同fit预算内试到并选对称MiniLM，L0四探索停在词法；证明该新任务下选择路线有价值，不是checked反馈/E5/组合收益，不外推普遍效果或按H调规则。评分0fit/API。报告batches/A130_A132_H1_paid_results_20261123_zh.md，证据research/a130_a132_H1_summary及a131/a132；当前A持有实际ledger且无活动队列，不再等已授权12条对账才付费，H2/Released后续继续。

## 2026-11-23：A125–A129完成接入、H1本地训练与真实账本接收

B5329503两份补丁已接隔离提交87634a5，CLI保留A守卫并接新调度，登记包装器修两处接口。82定向no-fit＋7默认菜单＋5训练前边界通过，失败保留；E5三文件SHA匹配，624.516秒34公共请求。H1原两outer本地L0各11fit，60.265秒模型封存不评分；匿名full35/主用新控制器真尝试E5等5路线，最终词法模型6fit，51.453秒，三种12行预测一致，未选ensemble。合28新fit/0API，A累计1566。用户B回复确认当前配置、924e88a下一轮；H2未跑B继续、无活动队列。

真实357账本附件已收到，SHA0aaedaaaaa7814ab9ad1f5f7e4bc7d3dd26faba260f936c5ec7abc829c5d4729匹配，旧335请求/503转移完整。当前账本runs/a129_B_ledger_return/private/ledger.sqlite，owner A/paid暂停，已知6.7114475/unknown12/pending0。新增Bean请求93c46fb2-af8f-4c1e-b2f3-cd019a6e2d68绑定HTTP200/286流事件但usage费用null，未解决不重试。旧335副本不用paid。报告batches/A125_A129_integrated_results_20261123_zh.md，主证据research/a125_a129_summary；H未开outer，三模型源与生产一致，旧C1/689保留来源。当前无活动训练/下载/API。

## 2026-11-23：B最新5329503和一次性交接说明

按用户指明采用B最新交付，fetch已到5329503；已读联合正式入口、单冲突、最新训练规模和两挑战者说明，生产能力未移植A。一次性交接A_TO_B_LATEST_INTEGRATION_SINGLE_HANDOFF_20261123_zh.md：A承担补丁适配/依赖/验收/固定实验快照，不让B停止开发或额外手写共同冻结回执；有现成主用命令就先接，不等待重复同意。真正需B补实际账本（不是JSON摘要）及新未知原始证据有无、H2现有状态与运行窗口；已交48/24/C1/输入不重催。原付费停止边界保留，本地接入照常。0新fit/API，A累计1538，无活动队列；这次是写交接文件，不宣称最新源码已经合并或H已启动。

## 2026-11-23：A120–A124当前源自主交付及离线验收完成

参数语义新模块默认off：40开发例36符合、错误矛盾0；四未通过为夹具未注册路线，旧4明确C错只检出1，其余未覆盖，保持未晋级。受控checked上下文+7226字节；74无fit定向回归通过、含训练测试1项排除。三个当前源任务从正式CLI自行选型/重训，每条11fit，三种独立交付2030/826/200行预测一致；旧Coupon包装失败后复用完整11fit，后22fit，总新增33，A累计1538。纯训练CLI合325.344秒，0API/新API费0。15交付故障通过，无新fit；所有包装/夹具/归档失败保留。H1工具prepare/train/seal/score可接续，训练和评分缺共同回执前置拒绝，H成绩0/未开留出。B新3347564摘要已看，方法/调度未合、真实账本/共同H冻结仍未回。报告batches/A120_A124_results_20261123_zh.md、交接A_TO_B_A124_HANDOFF_20261123_zh.md；主汇总research/a124_window_v2。当前无活动队列；本轮是五小时计划的可做项完成，不虚报连续运行满五小时。原C1为c64开发证据，最终同版本结果/Released/提交包尚待。

## 2026-11-23：下一轮五小时计划已落盘，尚未执行

计划见batches/A120_A124_next_5h_plan_20261123_zh.md：A120参数语义可选诊断，A121开发案例/历史回放与开销，A122三个已见任务当前源完整自主交付，A123条件满足后H1、否则离线交付/恢复与H runner，A124报告交接。每批先完成可复跑脚本并validate再运行。最多77新增fit/6Flash；未冻结或未收回实际357条账本并处理新增unknown则付费0。当前仅设计计划，新增fit/API0、A累计1505不变；没有启动训练或API，H共同冻结仍未完成。

## 2026-11-23：A118/A119五任务汇总与独立材料复核完成

c64五任务30终局/28完整，失败Bean两组保留。L1五任务与L0同预测、55→51fit；按原规则参考时间数值效率过线−42.79%，但两任务旧时间复用/A89约5倍波动导致效率证据不足，明确保留数值过线而不强制晋级。无质量方案达标，保守推荐L0，H共同冻结未完成。原矩阵281新fit+21复用/58Flash/已知¥.569714+1unknown是历史引用，本次0新fit/API，A累计1505。

48原例当前源45/48与B输出一致、错误确认0；3unknown边界，4accuracy构造争议保留原例，41/44仅敏感性。24单B Codex标签R1错/E5错；A后验重点复核4C方向明确错、1历史范围错、1术语歧义，不把标签当人类真值或重复前缀当独立错误率，不修改核验器追封存分。报告batches/A118_A119_five_task_and_independent_results_20261123_zh.md，证据research/a118_five_task_review、a119_independent_review；共同冻结建议A_TO_B_FSTAR_AND_H_FREEZE_PROPOSAL_20261123_zh.md。付费仍暂停等待实际357条原账本交回/新unknown核验；H未评分，无活动队列。

## 2026-11-23：A117收到并核对B补交C1/D3

HJL最新7f6458d，关键db589a3已含Task2/Bean六组、48原例、24标注。A按Git原始字节核对，12模型终局回执一致、10完整/2检查点，source匹配A原c64，24原回答逐条相同；45/48预期一致与标注6/8/10汇总可复算，但单Codex不算人工金标准、案例构造争议/语义尚待裁决。Task2五组668/940，E0为655；Bean完整四组1989/2178，R0/E1部分交付1986/1988。历史B112fit/22API、21有usage、已知¥.219394，新unknown1。交接摘要团队357/known6.7114475/unknown12/pending0，SQLite原件未回，A335副本过时，付费继续暂停。A本次0fit/API、累计1505；报告batches/A117_B_C1_D3_received_20261123_zh.md，验证research/a117_B_C1_D3_receipts。

## 2026-11-23：A112–A116原生L12整合与实际交付、H1准备完成

原生提交16dc903已通过deae3bb合入A主线，仅接standard/deep/deep-edit与资产/导出，保留A审查修复；每运行设/恢复目录防环境泄漏，组件控制器不接。15定向无fit检查、7默认任务目录通过。B声明L12原revision官方404原批保留；A113R用作者a50ef00143b4d5391434df20ae11632588ac25be取得三文件，大小/SHA全等于B原值，97.266秒ready。真实合成工程探针75.766秒，2native各primary+refit共4fit/0API，各12行原生/源码包一致，缺资产拒绝；A累计1505。不是自主性能证明。合并后source与探针/H入口逐文件相同，原c64隔离源码未改，回执research/a112_native_merge_verification。

A114保留A13原2000行PAWS范围，构造两训练侧1599/1600行、六新登记内部划分，8安全对称方法；B全49401行仅登记核对，不能混同范围。A115新native源码61.703秒，H1/H2共8真实本地CLI在Executor前截停/16 Flash参数通过，outer评分0。A116只读资源表：R/E均18请求/63fit，估计¥.139927→.210393(+50.36%)，不宣称核对省钱；去重28/48仅省重复fit需求。全队列新增4fit/0学校API，原C1不变，无活动队列。报告batches/A112_A116_queue_results_20261123_zh.md；未打开H性能或课程val/test，正式冻结仍需剩余C1/F*。

## 2026-11-23：A111接收B新强方法交付c658477

fetch已得到codex/b-verified-preference-1123=c658477，展示分支2b76156指向其发布副本。B原生L12/deep-edit标准Agent两组各10fit、0API，同为1224/1693(72.30%)；改写无增益。组件旧配对1208→1223、10→7fit，但候选和调度混杂、时间缓存不同。A独立checkout审查，2无fit作用域测试通过，7任务full/baseline共14默认菜单与A相同，已发布结果/用量/预测指纹元数据一致。本机未取得新权重/模型，不冒称重现其性能；A生产尚未合入。0新fit/API，A累计1501。当前c64 Task2/Bean六组增加0，F*未选；独立48/24对应的9ed包仍另待原件。报告batches/A111_B_verified_delivery_review_20261123_zh.md，证据research/a111_B_verified_intake。

## 2026-11-23：A110正式H2入口验收完成，尚未开性能

计划和代码先落盘，41.594秒完成4次本地CLI在分区完成/Executor前截停（两外层各L0/L1），真实worker/行号/标签指纹与B登记一致，full9。8份Flash参数只解析校验，未执行handler/资金预检；12份注册分区启动模板备好，正式H2仍L0/R0/F*去重。0fit/API/预测/留出图片访问，A累计1501。fetch后仍缺B新9ed提交/当前Task2-Bean结果，原五任务选F*和共同冻结尚未完成，故不提前用H跑性能。报告batches/A110_H2_entry_results_20261123_zh.md，脱敏证据research/a110_H2_official_entry；无活动队列。

## 2026-11-23：A107–A109解决A本机H2原包与训练输入准备

用户要求A自行解决。A107续传长流247.781秒IncompleteRead停止，保留48,234,496字节；A108按4MiB分段12次/324秒完成94,658,721字节原包，SHA/作者MD5通过，随后因Windows CRLF导致inner manifest原始SHA不同停止。两训练ZIP、六worker字段已一致；诊断LF后SHA精确等于B登记值。A109先落盘计划/脚本，固定LF离线重建，83.344秒prepared，原包只打开登记5000训练图，两ZIP/两manifest/六worker全部原始指纹与B一致，full9图像目录可用。0新增fit/API/预测/留出图片访问，A累计1501不变。所有失败保留，不改原B脚本或生产控制器。输入runs/a109_H2_canonical/training，原包runs/a108_H2_segmented/EuroSAT_RGB.zip；归档research/a109_H2_canonical_inputs，报告batches/A107_A109_H2_local_results_20261123_zh.md。无活动队列，H2性能仍待共同冻结后执行。

## 2026-11-23：A106接收B离线成果摘要，准备账本快照

B现已报告本地完成48案例和24标注，并表达Task2/Bean、H2、最终Task2承接意向；9ed50e6原提交和新bundle尚未收到。用户实际附件/同目录bundle仍为旧11月21日a787，fetch未找到新提交。结果状态更新为“摘要已收，原件待审”，不按未上传推断未做。A106 v2准备真实335条/known¥6.4920535/历史unknown11/pending0的一致性SQLite快照，私人文件忽略、源/副本全请求摘要相同，未交付/转移写入权。首次跨工作树ignore查询保守拒绝，原失败保留，修正在目标所属仓库验证后成功。0新增fit/API，A累计1501。回复与复核要求见B_REPLY_ACK_AND_NEXT_20261123_zh.md；无需重问24回答，H2优先B已有资产，adapter仅开发候选。

## 2026-11-23：A102 SMS六组完成、A103三任务汇总；A104登记可复用，A105原包获取停止

A10263新fit/12成功Flash/69748输入14880输出tokens/估计¥0.129268/1579.719秒，A累计1501，真账本335/known¥6.4920535/历史unknown11/pending0。SMS R0/E0 macro-F1 .945980，其余四组.936086，两类预测差15行，固定词法净多对3/826（多识别3条spam）。动态Flash内部两次晋级增益均值+.4620pp，却在外层−.9894pp；核对同原始结果相同。新进程交付与source一致通过，controller已退出释放睡眠请求。首次0fit预检分区差异和修订登记完整保留，六组实际使用同一新登记索引，不冒充旧A47。

A103只读汇总0fit/API：同source Coupon/花卉/SMS三任务E1−R1全0且同预测，E0−R0两平一负，动态质量/耗时有任务依赖；不晋级F*，还缺Task2/Bean同轮矩阵。A104原H2封存名单2外层/6内部/6000文件身份核对通过，无图片打开。A105在SMS完成后单次公开下载，300秒窗口后停（总310.625秒、partial30408704字节），未校验原包/未解码或构建输入，不自动重试。两批0fit/学校API，H性能未开。

报告batches/A102_A103_SMS_C1_results_20261123_zh.md、A103_three_task_conclusions_zh.md、A105_H2_preparation_stopped_zh.md；实际数据/模型/原包/账本均不提交。当前无活动任务，A暂停付费写入，等B回复协调承接/账本及已有H2原包，不以独立48/24阻断研发。

## 2026-11-23：A102继续；A104封存登记轻量复用核对完成

A102本地动态路径已完成10fit，首付费组3调用成功；最终以actual completion/ledger为准。A104只读B原source_ids压缩名单及已提交回执，2外层/6内部/6000独立文件身份、训练留出边界通过，无图片打开、特征提取或性能评分，新增fit/API0。A侧原RGB包/训练输入尚未重建，避免在当前计时矩阵运行重负载准备。A103只读macro-F1分析脚本已准备，不能在六模型冻结前读评分标签。

## 2026-11-23：A102独立旧源SMS六组已启动

用户授权不等B继续。计划/runner先推送b420177，独立c64生产worktree a102_sms_c1_checkout，输出runs/a102_sms_C1，最大66新fit/12Flash、A起点1438；A持有唯一真账本写入权。首次0fit/API预检因实际划分不同于旧A47停止并保留原始；新协议明确当前c64强环境实际索引重新登记（仍707/808/909），v2预检通过，六组同输入/指标/实际划分/16菜单/11fit/3调用/300及1200秒，原826开发留出统一在六模型冻结后评分。先看actual completion，不重复启动/改活动源码；独立48/24不阻塞，H/test继续封存。

## 2026-11-23：供用户转发的A/B对齐说明

已编写docs/AB_ALIGNMENT_REQUEST_20261123_zh.md，覆盖已有成果复用、原九天计划位置/四天研发窗口、三种源码用途、建议两人完整实验与交付分工、同数据/目录/预算、H封存、独立48/24协调、唯一账本与回执格式。请B先回现状与承接范围，不默认其已接新任务，不代用户发送消息。新增fit/API0，A累计1438。

## 2026-11-23：按原九天计划核对里程碑

实际处于D4整合/冻结准备，D5同源两任务六组部分提前完成，D3主要开发矩阵完成但独立48/24尾项未齐；D6冻结后H和D7最终Released尚未完成。用户剩余6天，其中4天研发、2天演讲；优先按原D4–D7补关键对照/冻结/H/交付，结构扩展仍受原预算与功能截止约束。完整表NINE_DAY_MILESTONE_STATUS_20261123_zh.md。只读核对及文档更新，新增fit/API0，A累计1438。

## 2026-11-23：A100–A101证据复用与候选冻结准备完成

新增fit/API/费用0，A累计1438。A100索引7组证据/19文件SHA，A89/A97同source，当前只差prose_claims；B16路径/170fit历史开发对照保留，不混作本轮六组。A101校验2套权重原大小/SHA、7种任务菜单和6组CLI模板；H仅登记元数据，未加载原样本。旧H2 Rice缺图片解码契约，编码器未启用，B的EuroSAT图片封存准备是可复用候选但尚未在A重建。两批passed/内部0.062及0.422秒，详情batches/A100_A101_evidence_freeze_results_20261123_zh.md。接下来优先A负责的单文本完整覆盖与H输入/版本对齐，独立48/24未交接不阻断；不直接给B派全量重跑。

## 2026-11-23：直接核对队友昨日GitHub进度

fetch无新B提交；直接读取主交付fa58e85及四研究分支。B确有Task2/Bean固定动态、Q与模型交付成果，本次未提交两任务完整六组成功结果（旧正式矩阵回执0/12）；独立48/24既定格式输入未找到。共享A29“两任务六组完成”属于A旧优惠券/标题，不是B的Bean。完整证据与复用判断见B_PROGRESS_VERIFICATION_20261123_zh.md，不默认让队友全量重跑。此轮只读核对，0fit/API，A累计1438不变。

## 2026-11-23：纠正计划与交接状态

用户说明没有要求队友执行此前提到的具体实验。Task2/Bean六组和独立48例有拟定计划依据，24回答独立语义标注为后续补充验证；但没有确认队友承接，不能把本机建议写成对方正在执行或欠交结果。CURRENT_AGENT_STATE、A94交接和D3独立验收说明已补明确状态。此轮仅核对计划与沟通状态，新增fit/API均0，A累计1438及账本不变。

## 2026-11-23收尾：A97六组完整结束；两类自然审查误报已修复

A97同9候选花卉六组均87%（174/200），最终CSV逐行完全相同；本地/原始/核对均选冻结MobileNet特征。动态组10fit、固定11fit，少1次并不始终更快。新增53fit＋10显式复用、12成功Flash、51230输入11559输出tokens、估计¥0.097466，1091秒；A累计1438。真账本323/known¥6.3627855/历史unknown11/pending0，实际扣费/总预算未核验。六模型新进程核验通过，source未变，controller退出并释放临时睡眠请求，无活动训练/API。报告batches/A97_flower_C1_results_20261123_zh.md。

A98在隔离工作树修历史baseline引用范围，16开发例6→16、Coupon原12提案机器误报1→0、47相关检查通过；A97结束后才合入。A99另补likely/unlikely to假设语法，10开发例5→10、花卉原12提案机器误报1→0，原16例仍通过，51相关检查通过。均0fit/API、原日志不覆盖；开发审查/软件回归不替代独立48/24语义验收，不声称新source端到端增分。

下一步B按c64bee4的生产内容与A94已验证实际行号，完成Task2/Bean整组C1；新HEAD含审查修复，不与旧source默默拼表。A暂停付费写账本，双方明确移交唯一原真账本后才换写入方。H/test封存，未见迁移和独立语义验收仍待完成。

## 2026-11-23：A96三组花卉Q完成，A97六组决策对照已登记

A96新32fit/300.438秒/0API，A累计1385；同800训练/200开发留出、真实三分区、11fit预算，baseline7与梯度无预训练8均42.5%，full9自主选冻结MobileNet特征87%，净多89/200（95增益、6损失）。原/无预训练模型及预测相同，三模型新进程重载通过。该队列B已开发过，只作已见真实对象任务的目录能力对照，不是未见H或Flash效果。详见batches/A96_flower_Q_results_20261123_zh.md，所有训练已结束，防待机上下文释放。

A97保持同9候选，比较本地/原始历史/核对历史×固定/动态六组；L1显式复用A96 full的10fit，其余五路径最多55新fit、12Flash。小批计划与可复跑脚本先落盘，预检0fit/API后才启动；当前实际状态读CURRENT_AGENT_STATE及独立completion。原ledger311/known¥6.2653195/历史unknown11/pending0不重建，新增估计阈值合计¥1.20，非扣费硬保证。

## 2026-11-23：六组收尾与真实花卉Q启动

A89完成53新fit/11复用/12成功Flash/56.8分钟；L0/L1/E0 75.0246%，R0/R1/E1 75.2217%，差4/2030，核对固定反而低于原始固定、核对动态与原始动态预测完全相同。新65966输入14405输出tokens、估计¥0.123586；ledger311/known¥6.2653195/历史11unknown/pending0，实际扣费未核验。A91完成10fit，A累计1353，两个临时护栏已释放。该旧photo实际Olivetti身份分类，后续不扩展、不写成自然花卉证据。

A94已生成三任务九分区实际行号/内容摘要，0fit/API，移交B的Task2与Bean整组实验。A95使用B已登记TensorFlow Flowers作者不交叉800/200、5花类64RGB，原包SHA验证通过，0fit/API。A96已启动像素7/加梯度8/加视觉9三组自主Q，最大33fit/0API，全模型冻结再统一评分，先查actual completion；具体CURRENT_AGENT_STATE.md。生产源码不改，教师已允许通用预训练特征且要求自主选用，H/test继续封存。

## A93老师说明已到，恢复通用下游特征资格；C1继续

用户已转来老师原文：限制针对SuperAgent语言模型，下游可用通用预训练特征，例如MiniLM，须自主选用、不得任务身份路由。TEACHER_ENCODER_CLARIFICATION_20261122_zh.md保留原文；A90未明确状态为历史，不再等待同一确认。wrapper默认auto，有校验资产时提供两编码器，真实full菜单15/32/16/9/5；无资产保留普通方法，restricted仍供消融。A93六入口检查/实际菜单快照0fit/API通过，核心及A89 immutable未改。

A89仍按既有Coupon六组队列继续，L1显式复用；A91排队full8不改其程序/seed/预算，研究角色转为禁用预训练视觉的消融，与A88 baseline7/full9比较。当前实际执行优先CURRENT_AGENT_STATE.md和completion/真ledger，AGENTS已清理为稳定约定，旧原文归档A92，不据旧阶段快照重复启动。

## A90资格边界更新；A89队列继续

课程“仅Qwen/DeepSeek”条款未明确冻结MiniLM/ImageNet例外，已问用户老师说明，尚未回复。默认环境入口改restricted，research需显式选；未改核心或活动A89。默认full菜单Coupon15/偏好30/SMS14/photo8/Bean5；A83旧研究full保留。编码器技术执行不认证课程资格/计价，A88 photo full尝试过视觉，暂作开发研究，不能因赢家梯度而省略该事实。A90 0fit/API的5入口检查及restricted菜单快照通过，详见A90_ENCODER_QUALIFICATION_BOUNDARY_zh.md。

A89仍在跑，L0已11fit/535.687秒，E0已11fit/3成功调用/857.406秒，原始动态后续进行；同Fields参数/硬预测相同而时耗5.12倍，根因未定位，复用L1旧秒数不作新效率因果证据。研究期间只写工具/文档，活动source未改。先读CURRENT_C1_A89.md与真实completion/ledger，不重复运行或把预算上限当实耗。

## 最新A88完成，A89继续Coupon六组完整对照

A88四模型全部冻结再统一评分，43fit/363.890秒/API0，A累计1290。Coupon74.8276→75.0246%（净+4/2030，106预测不同）；photo95→97.5%（净+2/80，5预测不同），梯度方法胜出、冻结视觉实际试过未胜出。仅两已见开发划分，详见batches/A88_A_track_catalogue_Q_results_zh.md，不把A47旧source分数拼作新对照。

接着按CURRENT_C1_A89.md与batches/A89_coupon_six_arm_C1_plan.md推进六组。L1预登记复用A88同源同预算11fit，其余5路径上限55新fit/12Flash/四组估计阈值合计¥1.20。统一全15菜单/实际split/300/1200，calibrated-v2，原真实ledger/incremental-research，历史11未知不清。先看actual completion/checkpoint，不把预算上限当实耗；任何请求或训练失败停，无重试。一个任务C1不代表跨任务矩阵完成，H/test封存。

## 当前A88已启动：A线两任务同源Q

代码/方案/脚本先提交aa47bb9并推送，再启动四条真实路径。先读CURRENT_Q_A88.md、batches/A88_A_track_catalogue_Q_plan.md。Coupon与photo各baseline/full，同一正式入口/指标/实际划分，最多44fit，trial300/path1200，0API；全部四模型冻结后统一开发留出评分。上批累计1247，本批按实际physical日志叠加，不把上限当实耗；H/test封存，不改活动源码或重复运行。下方A87“无活动任务”只对应上一阶段。

## 最新A74–A87已全部结束：强方法真实接入与交付完成，进入共同冻结

本对话33实际fit，A累计1247，新增学校API/费用0，无活动队列。读[batches/A75_A87_encoder_integration_results_zh.md](batches/A75_A87_encoder_integration_results_zh.md)及[A_TO_B_SHARED_RUNTIME_20261122_zh.md](A_TO_B_SHARED_RUNTIME_20261122_zh.md)。A74 14fit交付；A76 76项禁fit检查；A77 11fit正式本地证据提议/模型交付（A78只读纠正日志漏计5提议）。核心/非线性/组合已实合并，不再是review候选。

A75慢下载取消819.662秒；A79环境装好、旧revision404停295.071秒；A82新固定revision下四文件与原SHA全同/98.647秒；A81四真实编码器动作4fit通过，但fixture公开modality写错令CLI fit前停92.969秒；A84仅修fixture独立恢复4fit/102.922秒，自主试两编码器、选plain MiniLM、31行源码包CSV一致、缺权重拒绝。原失败不覆盖、不重复四worker。

A80导出v2外部权重/版本/预测前校验，3检查及旧包兼容通过。A85损坏副本硬链接权限失败保留；A86普通复制0fit/60.969秒，真实nonlinear/vision各8行跨进程一致，缺权重/同大小损坏拒绝。A83已记录实际菜单：Coupon15、偏好32、SMS16、照片9、Bean5；baseline分别14/22/6/7/5。强运行入口已准备，资产与venv不进Git。

A87 0fit/34.562秒：偏好全量单次编码粗估69–86秒，SMS22–32秒；现有worker有3次transform，不能拿小试秒数推全量。正式建议共同trial300/path1200秒，C1中心约2–4小时/名义路径上界10小时，均为预测。下一步冻结实际输入行号/source/菜单/环境/开关/预算后做Q/C1；不是正式矩阵已跑。H/test未开，独立48例/24语义标注仍缺，真ledger299/known¥6.1417335/11unknown/pending0未动。下方“准备中/未启动”等为历史快照。

## 最新A74–A82：核心与文本增量已实合并，编码器资产准备中

A74核心真实14fit/60.365秒、4路径正式新进程/源码包CSV一致；A76非线性/有界组合/canonical模块增量在独立checkout合并，76项禁fit兼容检查通过。A77新增正式开关--local-text-proposals evidence（默认off，local/ab-v1/full），实际10fit自主SMS路径+1pair探针=11fit/41.146秒；5次证据提议均实际完成，新进程CSV一致。统计脚本误读event/kind导致旧completion漏计，A78只读审计纠正，原产物/原runner保留。读batches/A74_merged_delivery_results_zh.md及batches/A76_A78_text_integration_results_zh.md。核心/文本已进入当前A，非仅review候选。

本对话新增25实际fit，A累计1239，学校API/新增费用0，旧ledger299/known¥6.1417335/unknown11/pending0保留。A75全环境公开依赖下载无代理过慢，核对自有pip后取消并保留失败；A79显式进程公共代理已成功安装锁定依赖，固定MiniLM旧revision返回404而停止（295.071秒）。A82按官方现有固定revision下载，仍要求B原全部SHA/字节一致，当前须查actual completion，不能视为ready或重复请求。学校API路由未改。

A80修导出包的外部权重与运行依赖声明/预测前校验，3项无fit检查与旧词法包兼容通过（10.837秒），不是实际编码器模型交付。A81真实编码器9fit上限计划与runner已保存，**尚未启动**；仅A82ready后用A79独立Python执行。H/test仍封存，D3独立48/24标注仍缺。下方A73“未合并”及更旧运行快照均已过时。

## 最新A73：B新交付已审阅，核心独立整合候选完成

已fetch并核对B五个开发分支及hjl交接入口，精确refs与无训练证据归档research/a73_b_upload_review，详情[B_UPLOAD_REVIEW_20261122_zh.md](B_UPLOAD_REVIEW_20261122_zh.md)。核心fa58e85与A67ba67f在独立checkout合并为9c7ceaf（codex/b-integration-review-20261122），当前A工作分支没有替换生产模块。64项禁网/禁fit兼容检查通过，保留首轮训练型测试被拦截的失败记录。A开发48例初验47/48，修研究适配器区分语义未知与执行拒绝后48/48；案例期望与生产守卫未放松。

B已有B1/B2开发结果、非线性冻结文本跨来源正面证据、有界反馈组合和偏好输出适配，可复用而不盲目重跑。独立实验分支尚未实际整合；其成绩本轮仅审阅未重训。本机缺torch/torchvision与已校验MiniLM/视觉权重，新增编码器候选未激活。优先有限预算匿名执行/新进程预测验收，再补合法运行资产、增量接强文本，冻结目录后做Q/C1/H。

本批实际fit/API/费用均0，A累计1214；原ledger299、known¥6.1417335、历史unknown11、pending0不变。独立48案例与24独立语义标注仍未在新交付中找到，D3不能宣布全面通过；H与正式test未开。

## 最新：A67–A72本轮收尾，D3主要开发矩阵已补齐

所有训练/API队列已结束；原A70评价失败保留，A71是独立恢复评价，不改旧completion。先读batches/A70_A72_feedback_and_review_results_zh.md和batches/A69_B1_coverage_results_zh.md。当前A累计1214拟合尝试，新增207＝A67回溯15+A68实测3+A69168+A70超时1+A7120；不能只统计完成模型。API新增24成功、80083输入/21268输出、已知折算¥.165155。ledger299/known¥6.1417335/历史未知11/pending0，实际扣费及总预算未核验。

A69四任务×两内部seed×固定/动态16完整路径、168fit/879.761秒，8对0涨7平1负，动态少8fit，Bean1707−.0918pp；使用固定的已登记外层val，不是两独立外层。后验同10条测量仅换到达顺序可换赢家，表明原序列晋级有路径依赖，不是新策略效果证明。

A70四任务本地前三步12状态，24个新Flash提案451.177秒全部封存；首个评价60秒超时1fit，原queue停。A71同源/同回答/同候选/同seed统一180秒评价，20去重fit完成48逻辑评价，101.039秒，0新API。12对10平2负0胜；Coupon/Bean0，偏好文本平均−.2007pp，SMS macro-F1平均−.4049pp。没有普遍反馈收益，不追seed凑正面。自然结构化错误机会0，纠错成功率仍不可测。

A72后验发现review两条历史对象误报，修复通用引用范围：未绑定历史回顾不默认指当前候选。原machine flag2→0，同时5原确认转未知；40项相关无训练检查和48开发例通过，24原A70上下文在新代码下重建相同。源码已更新，旧开发性能仍按原source记录，不拼作最终C1。

D3训练探针、A侧48开发例、B1修订四任务矩阵、A2修订四任务机制实验已具备产物；**独立48案例来源与24回答独立语义标注尚缺**。已生成research/a70_independent_review/blinded_reviews.json，隐藏显式R/E但措辞可透露条件；不给机器标签作真值。转交docs/D3_REMAINING_INDEPENDENT_REVIEW_TO_B_zh.md即可，格式/runner齐备。H/test仍封存。D4先整合强方法与统一版本，不强行晋级更差的反馈/动态默认。

## A69 B1开发覆盖已完成，A70真实提案正在运行

A69四任务×两套内部划分×固定/动态16路径全部完成，168实际fit/879.761秒/API0，16模型冻结后评分并新进程预测通过。8对0涨/7平/1负；动态80对固定88fit、390.844对429.500进程秒，Bean1707少对2/2178（−0.0918pp）；7对预测完全相同。只用一个已登记外层val，不能说两个独立外层复验。A69后A累计1193；A70训练另计。

后验0fit/API同证据重排：固定Bean1707同样10条实际测量仅换合法到达顺序，赢家C1.0→C0.1；原序列晋级有路径依赖，并非新策略效果证明。结果batches/A69_B1_coverage_results_zh.md。A70已成功冻结4任务12状态，正在一次性24请求上限的ask，所有回答封存前不训练动作。先查actual阶段completion/真ledger，不重复计费，生产与活动脚本仍冻结。

## A67/A68完成，A69→A70已启动（以completion核对实时状态）

先读CURRENT_D3_A67_A70.md。A侧统一48开发例，旧源44/48，有4例跨split/payload/primary metric错误确认；已加comparison_scope和正式AB观测身份，修复集48/48。旧缺provenance入口不称完整核验；独立B48尚缺。A68正式匿名两观测+refit共3fit/9.624秒通过，5受控scope冲突全拒绝，新进程predict通过。最终52无训练检查通过。

资源修正：阶段55项回归包含HypothesisAgent.IntegrationTests真实Dummy.fit共15次，依据未改测试源码/通过日志回溯补记（非原物理观察），不能记0。A66后1007+15+3=1025，API0。research/a67_cases/regression_fit_correction.json保存依据。

A69预登记四任务两训练内seed固定/动态16路径、最多176fit/API0，已实跑；source/目录/实际split全批冻结，16模型全封存再评分。A70已启动等待A69成功，再复用本地前三步prepare12状态、最多24Flash/48去重fit，唯一真账本/incremental-research，11历史unknown保留。独立输出、不自动重试；代码与计划均已保存。当前不把计划上限当完成量，H/test未开，不修改活动源码或重复启动。

## 最新A66：A1八格真实训练输入探针已补齐

起点c1514db，执行前计划batches/A66_semantic_training_matrix_plan.md、runner与观察器已保存。preference/symmetric × primary/full-refit × seed19411/19421八格全部实际训练通过；捕获分类器真实特征/y，preference正确翻倍+翻转，symmetric不复制不翻转；训练内词表/查询隔离与交换概率验收通过。另8次相反语义候选在fit前拒绝，未多训练。

26.478秒，新增8实际fit/API0/费用0，A累计999→1007，无运行任务；原1000为建议线，8次按本批有限验收预算单独登记。生产源码未改，H/test未打开；合成worker性质探针不代表8条完整自主Agent或真实迁移。结果batches/A66_semantic_training_matrix_results_zh.md，脱敏research/a66_semantic_matrix。D3清单训练探针项可勾完成，96案例/独立封存、A2语义与范围、B1矩阵仍未完成。

## 最新：A64/A65 D3验收补充已完成

用户授权继续未完成D3。清单docs/D3_ACCEPTANCE_CHECKLIST_20261122_zh.md（本目录下同名文件）、结果batches/A64_A65_results_zh.md。当前源76已知回归全通过/7.464秒，新增fit/API均0，累计仍999。旧预留案例已看过，不冒充新封存48例。

Git fetch成功，origin队友d750c7b/main a38e1b8未更新，a787a28本机交接包已接；不代表队友本机无新工作。D3原A1 96例/独立封存与8格探针、A2四任务统一矩阵/语义纠错、B1四任务两划分仍需补齐或明确范围修订。

A65六状态12回答机械约束全部有效，数值预期全null；结构记录中的失败/配置矛盾/不支持假设机会均0，纠错率不可测。文本三状态局部E−R均值+2.40899pp，Coupon0，为已有A55测量再分析，不新增独立证据。生产代码未改、H/test未开、无训练/API运行。1000fit旧建议不能当学校硬限制；下批先登记有限预算再推进，优先完整验收和正式证据链。

## 最新：A60–A63完成，进入D4整合准备

- 先读[batches/A60_A63_results_zh.md](batches/A60_A63_results_zh.md)，下一步[D4_freeze_and_formal_budget_20261122_zh.md](D4_freeze_and_formal_budget_20261122_zh.md)，队友交接[A_TO_B_AUC_FINALISTS_20261122.md](A_TO_B_AUC_FINALISTS_20261122.md)。没有训练/API任务在运行；下方A37–A59为历史。
- A60原动态/统一finalists两任务4完整路径，26fit/187.641秒/0API。Coupon均75.0246%；偏好69.5745%→69.6809%（+0.1064pp、多对1条）。统一组两候选都有同一confirmation证据，但探索分配也变，不能单独归因或宣称稳定涨分。7fit小试，未替换默认S。
- A61五类真实旧模型概率与类列序诊断、人工硬标签同/AUC异反例通过，0fit/API。A62正式AB AUC在已见SMS显式换目标跑通3fit/15.525秒，开发AUC .991983/accuracy .881356，新进程概率核验/B导出通过；不是对原macro-F1的效果提升证明，多分类/Flash-AUC待测。
- A63合法AUC partial通过、六损坏反例拒绝（含硬标签相同而概率变），完整/partial两源码包独立进程及已有独立venv概率CSV一致，0fit/API。52个不同无训练契约/回归检查通过。
- 本轮新增29实际fit/API0/费用0，A累计999；原真账本275行、known¥5.9765785/11历史unknown/pending0的历史状态未清空。本轮禁网，本机账本未被替换；课程总预算仍未核验。原1000为建议线，先停止开发轮训练；正式阶段另按两人产物核算预留，不能把已花研发冒充正式Q/C1/H。
- A60与A62有各自冻结source，后者含AUC与防御性证据缺失检查；正式冻结版仍待两线合并。H1/H2与课程test未打开。脱敏统计归档research/a60_a63_finalists_auc。

## 最新完成状态：A37–A59本轮收尾

所有训练/API队列已完成。结果总表见[batches/A37_A59_round_results_zh.md](batches/A37_A59_round_results_zh.md)，后续计划见[NEXT_RESEARCH_AFTER_A58_zh.md](NEXT_RESEARCH_AFTER_A58_zh.md)。下方“运行中”均为历史快照，不可据此重跑。

- 本轮新增373次实际fit，71次成功真实聊天及1条经证据审计未发出的请求；330073输入/67439输出tokens，已知增量估计¥0.599829。累计A窗口970fit；真账本275行、known ¥5.9765785、原11 unknown、pending0；平台扣费未知，课程总预算未核验。
- A50：训练事实×可校准提案2×2，两任务85fit/24调用/36.65分钟；联合组与原始组外层成绩相同，尚无稳定准确率收益。A51跨SMS/图像联合组也持平，32fit/12调用。12份v2提案均未报数值预期，不能称预测校准已改善。
- A55同历史六状态重新比较：12调用/18fit，偏好文本三个状态E局部收益2.20–2.59个百分点，Coupon持平。A58完整E闭环10fit/3调用/113.735秒，偏好文本69.5745%低于已有R的70.1064%，下降0.5319个百分点；复用R参考且仅一次，不作普遍因果判断。
- A59发现最终候选复核标准不同；主划分配对差异bootstrap区间为正，但不能覆盖训练方差/选择偏差。下一轮优先同一复核门槛与预算预留，再扩大方法与AUC任务。H1/H2、正式test未打开。
- A54/A57六个模型包在独立进程及全新依赖环境预测逐字节一致；124项最终回归与21项契约检查通过。A56当前A/B状态接口与损坏证据拒绝验收完成。

当前总计划：nine_day_two_track_plan_20261120_zh.md。两位成员均用 Codex，A 管任务全流程/证据，B 管方法与搜索。

## 每次继续的约定

1. 先读 AGENTS.md、总计划和本记录，核对 Git 与已完成产物。
2. 每小批先写 docs/batches/<编号>_plan.md 与可复跑脚本，写明资源、对照、验收和输出；然后才运行。
3. 逐批落盘 protocol/results/completion；失败保留，不覆盖。API 失败停止，不自动重试。
4. 每次对话报告：瓶颈/结构、做了什么、结果支持范围、耗时/拟合/API、下一步。区分计划、运行中、完成。

## 当前进度：A 线三小时计划内批次已推进（2026-11-21）

### 2026-11-22最新：A44–A49已完成；A50/A51真实队列运行中

- 接手优先docs/CURRENT_RUN_20261122_A50.md。A32被Windows00:21待机/08:32唤醒中断，8完整/1部分/11未启动，91fit/20成功聊天/¥.163567；未开外层val，旧completion保留stopped。三小时原窗口没有完整执行，不把待机九小时算训练。
- A42补充批首条1fit后沙箱DNS失败、未重试，原ticket审计not_dispatched保留unknown转移；原11历史unknown仍在。进一步对照发现两层：沙箱拒绝校园网；授权环境Python拿到198.18.*虚拟IP。显式进程解析官方域名到既有10.121.10.250后，带认证models200/19模型，Host/SNI/证书不变，系统代理/DNS未改。普通GET超时与成功TCP探测不能混称API已可用；a50_models_direct才是JSON模型核验。
- 新正式开关：task-profile train给raw/E同一训练统计；proposal-contract calibrated-v2允许purpose/uncertainty及null或下降预期；partial-delivery checkpoint异常后只包装已成功模型，仍failed/exit2。默认off/strict-v1保留旧行为，未改B方法/调度。
- A44–A46五真实开发任务十路径通过：5完整3fit、5人工故障2fit，共25fit/142.736秒/0API，10模型新进程搬迁预测一致，外层标签未开。两次失败验收3+5fit另保留，修摘要变量覆盖和checkpoint.kind误判；1合法+6损坏证据恢复探针0fit通过。另124回归7fit已计，最终版本回归待训练队列结束串行做。
- A47五任务同源Q全部完成，7物理路径/67fit/328.799秒/0API。Coupon目录14→15，accuracy .738916→.750246（+1.133pp、10→11fits）；照片7→8，.975→.975（10→9fits、3行预测不同）；偏好22/22、SMS6/6、Bean5/5目录相同，各一shared路径，不是独立重复实验/0收益证明。七模型重载通过，H/test封存。research/a47_catalog_Q。
- A48真实CLI与模拟SSE的v2/profile闭环3用例，9fit/5模拟HTTP/26.946秒、0真API，实际共享账本SHA不变；raw/E全部共同payload相同，null检查not_stated，模拟timeout后partial且不重试。模拟ledger/usage不导入真账本。A49参数观察独立原型五模型、五错误参数声明被识别/0自然冲突/0fit/API，尚未接生产。
- A50/A51计划batches/A50_A51_live_facts_calibration_plan.md与runner/queue已保存并实际启动。A50两真实任务2×2 off/train×strict/calibrated，8路径；首coupon_00已完整11fit/3真实响应/499.411秒、refit成功，尚未开本批val。后续A51仅在A50成功且余≥50分钟才启动SMS/照片联合对照，不把计划当已完成。截止恢复窗口10:42:29。不能修改冻结生产/执行器脚本或并发训练。
- A50前被跟踪A窗口812fit（含失败/回归/原历史40更正），ledger224行/known5.5403165/原11unknown/pending0。A50开始后必须看真ledger与completion，不能继续用此快照发请求。课程总预算仍未核验、实际平台扣费null；新API确已返回有效Flash响应与usage，当前不再以学校网络不可用为结论。
- A52四模型搬迁与A53首动作两新训练内划分计划/代码已保存，尚未执行；H1/H2仍未开。原A34/A35/A36/A38不能由新编号顶替成已完成。

### 最新A29/A30已完成；新三小时窗口A31通过、A32准备就绪

- A29两任务六组12路径全部完成，128实际fit/24成功请求/2459.935秒/¥.205921，132821输入18275输出；source不变、12模型新进程预测重载通过，H/test未开。各组结果见batches/A29_A30_results.md：Coupon L0/L1/R1/E1 .750246，R0/E0 .748276；偏好对L0/L1 .696809、R0/E0 .680851、R1 .701064、E1 .695745。单开发划分，无显著性或普遍迁移声明。
- A30严格首次公共历史相同，raw与checked各自两次完整messages相同；偏好对checked首次候选却不同，不能把差异全归因于反馈。3唯一动作×2新划分=6fit/0API/34.140秒，16逻辑评价复用10；Coupon首选全部同，偏好对word_char C.05较word C.1新划分均值+1.7772个百分点，但完整E1最终低于R1，局部与端到端结论分开。初次分析hash用错工具已修，0fit/API、原错误保留。A窗口累计597fit。
- 用户新授权继续3小时：北京时间11月21日23:24–22日02:24，计划batches/A31_A35_three_hour_window.md；本机runs/three_hour_20261121_2324/window.json记录起点ledger203/known5.3767495/11unknown/pending0。
- A31新增默认off的Flash候选路线投影，min(4,注册路线数)，仅从原eligible里优先未见路线，不改B调度/anchor/confirm/stop、不重问。124回归通过，7监督fit/0API；旧固定Flash12提案有10个在原状态下会被新门槛排除，0fit/API回放不冒充新效果。新窗口当前7fit，A窗口604。
- A32新源训练包/15与22候选/原实际划分重新冻结，20路径预检0fit/API通过；已启动a32_coverage_matrix，coupon L1 11fit/0请求、R0_off 11fit/3请求已封存，正接首个routes组。计划batches/A32_route_coverage_matrix_plan.md；最多220fit/48请求，截止02:24前不足660秒不启动新路径。以checkpoint/completion实时状态为准，不能把中途快照当完成或A29旧源当新代码控制。
- A31另将24个原始R/E请求在新代码off模式重建，内容逐项一致，0fit/API。A34四任务数据/划分prepare完成0fit/API；A34跨输入、A35复用R的固定状态、条件A36紧预算计划/代码与串行队列已保存，队列待A32完成再开新训练，见batches/A31_A35_three_hour_window.md。是否实际启动取决于余时/前批成功，不能将准备当已执行。

### A29/A30先前运行快照（现已完成，以上方为准）

- 用户要求连续推进。计划batches/A29_limited_choice_matrix_plan.md、runner scripts/run_a29_choice_matrix.py、独立worker观察器已落盘。
- a29_choice_prepared重新冻结coupon15/preference22候选、只含训练成员的包、实际707/808/909划分和当前生产源码。准备6.234秒；a29_choice_preflight验证12路径命令/同一累计账本通过，0fit/请求。
- a29_choice_matrix已启动：coupon L1 11fit/0请求/250.745秒、R0 11fit/3请求/414.426秒、E1 10fit/3请求/328.508秒已封存；尚未打开外层val。后续自动接L0/E0/R1及preference六条。实际树模型耗时较早期小试慢，中心估计更新约40–60分钟，原600秒/worker60/预算/顺序不变。当前运行状态以checkpoint/completion为准，勿重复收费请求或覆盖目录，不能把此中途快照当最终结果。
- A30预登记batches/A30_first_choice_audit_plan.md与runner scripts/run_a30_first_choice_audit.py已保存：复用A29每任务四个首提案，严格检查去checks/memory后的共同历史一致，0新增API；A29完成后在17171/17181两新训练内划分测不同动作，最多16fit、相同动作去重。当前只验收“主批未完成则拒绝训练”保护，0fit/请求，未跑动作质量评价。不得并发训练影响A29耗时。

### 最新A28：Flash网络恢复，真实SMS raw/checked两臂完成

- 用户网络调整后新授权；连通1请求返回ok:true/有效usage，7.597秒、¥.000136。随后按预登记正式CLI执行两臂，无改生产代码/网络、无重试。实际源码同202a250且两臂source_unchanged，H/test未开。
- SMS两臂首次请求去checks/memory后公共上下文相同，checked确实收到checks/memory；后续历史分叉。raw/checked最终均single_word_char_C3.0，826行外层macro-F1 .9459801186、逐行预测完全相同，两模型新进程重载导出通过；没有本批搬迁验证。
- 各7fit＝6探索+1refit、3真实Flash，未触发confirm。raw136.101秒/6048输入3211输出/¥.018892；checked133.352秒/13638输入2595输出/¥.024018；两臂墙钟278.569秒。含连通全批7请求/14监督fit/19754输入5823输出/¥.043046，0新unknown/pending，响应ID无重复。累计A窗口463fit、模拟HTTP仍25。
- 有真实“假设未支持→下轮读取下降→保留原赢家”的轨迹，但两臂都跑完6个候选，选择主要影响顺序，不能由这次平局断言反馈无效，也不能证明checked增益/未见迁移。下一批优先候选数大于探索次数的15/22候选任务，再按同版跑local/raw/checked，先落计划/runner；不追SMS seed。另记录正向涨分期望契约可能给诊断动作强配故事的限制，本批未改schema。
- 当前ledger179行/known5.1708285元/原11历史unknown/pending0/完整费用null；研究模式课程预算仍未核验。学校API现已实际成功，不再等账单/排网络。结果batches/A28_flash_network_resume_results.md、证据research/a28_flash_network_resume、分析scripts/summarize_a28_flash.py。完整Q/A2/六组C1/H/Released仍待后续。

### 最新A26–A27：A已补对称方法，Flash新尝试仍DNS失败

- 用户授权A直接补齐；新增训练内共享TF-IDF的绝对差+乘积/逻辑回归，word与word_char各C=.3/3共4候选，每评估/refit均1监督fit。AB目录只向明确symmetric开放；原偏好目录隔离，按实际已知执行函数认证preserves，不凭元数据flag。未改B调度。
- 最终116项通过3.517秒，7监督fit；匿名正式CLI自主4探索+1refit=5fit，CLI8.570秒/整批15.454秒，原/搬迁/交换输入三个新进程CSV24行一致、概率交换差0。仅合成工程性质，非真实语义性能。只读封存注册Spec，H1合法候选0→4、H2仍8，H原包未开、未训练/评分。
- 两轮失败保留：首轮旧测试仍期望空目录，7fit；第二轮观察器len(sparse)错误，3实际fit；最终7fit。修旧A09统计后93项再验收通过4.329秒、4fit。本批监督fit26＝7+3+7+5+4，回归21/CLI5。此前10轮schema回归硬编码0实漏40，按旧成功日志+未变代码回溯补记，原文件不改；A窗口383+40+26=449（非全项目/队友总数）。预处理次数未完整计数。上批103修为119含其4轮漏记16，A20性能102物理fit不变。
- A27零训练单次真实SSE测试：初版scope预检0请求拒绝后修正sms_spam，预检通过；真实1attempt约1.515秒DNS NameResolutionError/getaddrinfo11001失败，无HTTP/流数据、0模型请求发出/响应，无重试。只读诊断6.102秒DNS失败、校园IP443超时，不改网络。
- 原请求ticket/run/hash证明未发出后追加审计、保留原unknown链；当前ledger172行、known5.1277825元、原11unknown、pending0、完整费用null。显式研究模式继续，不等待平台账单；下一次只能在新目录按用户明确选择做测试，不能自动反复请求。
- 结果batches/A26_A27_results.md；用户测试CAMPUS_FLASH_A27_FAILURE_AND_TEST_20261121.md；B交接A_TO_B_A26_SYMMETRIC_20261121.md；证据research/a26_a27_methods_and_flash。剩真实Flash闭环/A2、剩余Q、最终C1/H/Released，不再把“等B补H1词法基线”列作依赖。

### 最新A20–A25：五任务本地C1已跑，不再等待学校历史记录

- 用户无法取得平台记录且明确“不用等，继续推进”。新增显式`--history-policy incremental-research`，保留历史unknown；默认strict仍拦截。新增unknown/pending立即停，结果标记课程总预算未核验/非预算合规正式报告。金额是估计预留与结算阈值，不是最终扣费硬上界；不承诺总费用已知。
- A22先补旧偏好文本对CSV/列契约，87检查通过、0分类器fit；随后a22_c1_readiness冻结。A20五任务十路径完整CLI，先全冻结再评分，原正式test未打开；102 attempted=completed fits、478.379秒、API0，十个模型新进程预测与重载评分通过。
- Coupon accuracy .750246、偏好对accuracy .696809、DryBean accuracy .913223、SMS macro-F1 .945980、照片accuracy .975；每对同方法同预测。固定55fit→动态47fit，少8（14.55%）；CLI合计226.830→197.312秒。只支持开发单划分节省训练，无精度或LLM收益，不是最终冻结六组C1。
- A21本地审计未恢复11未知的任何receipt。A23导入170真实历史、known 5.1277825元、11unknown，排除模拟，完整费用仍null。真实小试确实启动：1基线fit后第一请求DNS NameResolutionError，无HTTP/流数据，失败即停、checked未跑、留出未评分；原子路径保留3.568秒记录。
- A24只读网络诊断6.078秒，DNS失败、校园IP443超时，0聊天，不改网络。A25原请求ticket/run/hash证明未发出，追加审计而保留原unknown转移；当前171行、known仍5.1277825、原11unknown ID全保留、pending0、完整费用null。1真实连接attempt，0模型请求发出/响应；不自动重试。
- 最终93项回归通过4.306秒，0监督fit/API；首轮91项测试异常类型预期错误已修，原失败保留。四轮整套检查另4预处理fit。本轮103监督fits，A窗口累计383；模拟HTTP25未增。
- 结果batches/A20_A25_results.md；research/a20_local_c1与research/a21_a25_progress。续跑先读该结果与A_TO_B_A20_A25_UPDATE_20261121.md；学校网络可达后新目录按研究模式跑小试，不再等待平台账单。剩余真实Flash/A2、B合法对称方法/剩余Q、最终六组C1/H/Released仍待，H1/H2保持封存。下方早期“必须等11未知对账才允许所有真实研究”已由本次显式用户指令更新，不删历史记录。

### 最新A17–A19：照片Q对照完成，C1五任务准备齐

- A17同版新增--catalog-scope baseline/full（full默认），A执行器投影B目录，B方法/调度不改。照片7vs8候选、相同实际划分/11fit上限，2路径全封存后评分；两组均78/80=97.5%，不同3行（各独有正确1），无分数收益。baseline像素logistic10fits/22.746秒，full梯度9fits/18.488秒；两模型搬迁预测各自一致，整批19fits/56.029秒/API0。
- DryBean/SMS本机两目录逐项相同，标去重、不重复训练。完整五任务最终Q尚待B子矩阵/版本冻结，不以零增益追seed。
- A18发现课程旧适配器漏了README中的metric/标签含义，A19按唯一明示声明保守解析并保留来源。两加载入口一致，否定/冲突/多README/反向标签不猜，manifest优先；不是任务名路由。Task1/Task2目标确认，Task2从0到22个合法候选可见（未新增训练/方法）。
- 最终C1准备runs/a18_c1_readiness_current：五任务full目录15/22/5/6/8候选，30配置；10local待执行、20Flash仍因账本blocked。实际行ID/源/依赖/README来源齐，formal_experiments_completed=0。费用scope修为task1/task2/dry_bean/sms_spam/olivetti_faces，旧*-dev配置作废且未产生真实API；A16默认同步sms_spam。
- 初始81项catalog检查、三轮随代码补充的README86项回归通过，均0分类器fit/HTTP，共4预处理fit。本批新增19监督fits，A窗口累计280；模拟HTTP25未增，真实API0。Q保留修README前源码指纹，不冒称最终版Q。
- 结果batches/A17_A19_progress_results.md；最终紧凑证据research/a17_a19_progress_final，初版归档保留于本机runs/a17_a19_archive_initial。计划/runner齐备。H1/H2未训练未评分；真实11未知未对账；B合法对称方法、完整Q/C1/H/Released仍待后续。

### 最新：本次连续推进A12–A16接续交付完成

- A12：正式六组入口local/raw/checked×ab-fixed/ab-v1已接，ab-history-v1显式划分，三请求上限/失败停/累计账本；4成功+3故障场景通过，47fits/15模拟HTTP/0真实API，106.462秒，4模型两新进程含搬迁预测31行一致。R/E公共上下文一致，非真实Flash效果证据。
- A13照片：AT&T/Olivetti真实PNG文件路径8探索+1refit=9fits，25.173秒，梯度候选自主胜出，外层78/80=97.5%，模型搬迁预测一致；闭集小样本人像，仅开发证据。
- H1/H2数据与实际两划分完成登记和CLI任务打包，0fit/未评分。H1 PAWS-Wiki publisher train抽完整连通组2000行（1105/895），合法B方法0待接；H2 UCI Rice120张/3类/32RGB，8方法可见。H任务包runs/a13_holdout_packages，原文/图像不入Git。原Google403与慢下载已保留，明确修订官方publisher来源并核验摘要。
- A14对称输入equivalent标签契约修复，错误left声明训练前拒绝；25检查通过。首轮旧Dummy夹具误报0fit，保留原记录并更正1fit，计数重跑又1fit，两次累计2，不作性能实验。
- A15最终78项无监督训练回归通过，4.105秒，0分类器fit/HTTP；一次预处理fit（此前77项一轮另1次）。项目任一历史unknown/pending不能用新task ID绕过；本机11条真实历史仍未对账。
- A16真实调用小试runner/预登记就绪（22fits/6请求上限，默认仅校验），用unknown模拟账本预检正确0fit/0请求阻断；没有真实运行/没有清空账本。
- 本次连续推进新增58监督fits（47+9+2），A窗口累计261；模拟HTTP新增15累计25；真实API0。各批保留运行时源码指纹，后来的schema/入口门槛变更不冒称已经重跑旧模型。
- 结果docs/batches/A12_A16_day_results.md；模式/命令/队友接续AB_RUN_MODES_20261121.md；紧凑research/a12_a16_day。今日A线接入与验证范围完成；整个项目Q/C1/H/Released仍有后续工作。

### 最新 A-11：同版本固定/动态公平对照完成

- A正式CLI新增ab-fixed：4探索、primary前3各2次复核、1重训；探索函数/目录/最终选型与ab-v1动态共用。42项运行前接口检查通过，0fit。
- 三任务六路径，同源码、实际行ID、目录、预算、探索前缀均核验；全封存后评分。DryBean0.913223、SMS macro-F1 0.945980、digits0.986111，三任务两组预测逐行相同，无精度收益。
- 固定33fits（12探索18复核3refit）；动态27（15探索9复核3refit），省6/18.2%。CLI总83.270→68.452秒，仅单次时间描述。整批202.704秒，60attempted=completed真实fits，独立观察器一致，0API/0模拟HTTP，模型搬迁预测全一致。
- 当前A窗口累计203fits、模拟HTTP10；历史/队友另计。原始runs/a11_schedule_comparison；紧凑research/a11_schedule_comparison；预登记/结果docs/batches/A11_fixed_dynamic_*；给B A_TO_B_A11_UPDATE_20261121.md。
- 证据只支持已见开发划分省训练，不证明LLM、未见迁移或最终C1。下一步A接动态Flash raw/checked回调及实际账本保护（先无真实调用），B强方法/合法对称文本，真实自然图像与H登记仍待。11条未知请求继续未对账。

### 最新 A-10：真实三任务动态入口小试通过

- 生产基线07884d0，未修改生产代码。复用已冻结A07数据包；预先登记三任务实际行ID、目录与依赖，运行后逐项一致，三模型封存后统一外层评分。
- DryBean accuracy 0.913223（2178行），8fits/17.710秒CLI；SMS macro_f1 0.945980（826行），9fits/27.035秒；digits accuracy 0.986111（360行），10fits/19.266秒。各模型搬迁后新进程预测CSV一致，交付验证共16.536秒。
- 全批90.096秒、27attempted=completed fits，独立观察器同为27，0真实API、0未知训练失败。当前A窗口累计143fits；模拟HTTP仍为10。历史/队友用量另计。
- SMS在两共同划分后将word_char C0.3换为C3.0；digits允许primary稍差但两次复核更好的树替换logistic。梯度图像方法第2步实际训练但未胜出。不是固定/动态调度或LLM的因果收益证据。
- 协议A10_real_task_pilot_plan.md；runner scripts/run_a10_real_task_pilot.py；结果A10_real_task_pilot_results.md；紧凑证据research/a10_real_task_pilot。下一项优先同版本/同候选/同预算固定与动态对照。Flash接入、自然图像和对称文本H支持仍待；11条未知API仍未对账。

### 最新 A-09：B bundle 已合入，正式本地动态入口验收通过

- B a787a28 快进接收；新增正式 `superagent run --scheduler ab-v1 --portfolio ab-v1 --fit-limit 11`，接 A 状态/执行/交付与 B 目录/调度，不修改 B 选型规则。worker 正式加入 fit 前限额与持久计数。
- 匿名路径 5 explore＋4 confirm＋1 refit＝10真实fits，18.791秒；两个新进程含模型目录搬迁，31行预测一致。初次外层验收误要求B的恰好11次而失败，失败保留；改为按原约定“最多11”复验既有模型，新增0fit/3.754秒。生产源码未变。
- 两机依赖与实际划分指纹不同；只有生成配置/种子相同，不能称同划分公平对照或性能复现。详细说明 `batches/A09_B_bridge_cli_results.md`。
- 38项接口回归通过（3.519秒，0fit）；旧CLI五项通过（29.628秒，17fits/5模拟HTTP）；本批合计27fits/0真实API。本A窗口累计116fits、10模拟HTTP；历史与队友用量另计。
- 协议/runner/紧凑证据齐备：A09_B_bridge_cli_plan.md、scripts/run_a09_bridge_cli.py、research/a09_bridge_cli；给队友 `A_TO_B_BRIDGE_ACCEPTED_20261121.md`。
- 动态CLI尚未接Flash raw/checked；AUC分数证据、对称/关系文本候选仍缺。下一批做真实任务完整小试与环境/划分对齐。11条未知API仍未对账。B合并改变源码，A2旧冻结状态仅作历史，不再声称与新版同源。

- 用户明确此前做 B、现在切换 A；历史文件编号保留，我们当前只主改 A 的任务/证据/入口/交付，队友主改 B 方法与调度。
- 已发布共同基线 `0b112d823a1991686281fdc88602e758e29a4202`，远端分支 `codex/expanded-feedback`。
- 计划：`batches/A_next_3h_20261121_plan.md`；结果：`batches/A_window_20261121_results.md`。这是计划内批次实际成果，不宣称已经持续运行满三小时。
- A-05 开发18/24→24/24，预留10/12→12/12；A-06 4/12→12/12；A-07五真实路径完成、35fits/97.21秒；A-08边界28项通过，真实B调度尚未接入。
- 目标来源可追踪，完整指标/正类身份核对，导出前行/类/概率校验已实现；导出故障2/9→9/9，三个旧模型经最终导出器重载一致，无新增训练或外层重评分。
- 50项受影响回归通过，21fits；最终CLI五项通过，17fits/5模拟HTTP；本窗口累计73真实fits，真实API0，11条未知历史请求仍未对账。
- history-evidence-v3已更新并检查两组共享字段一致，旧A2状态不能冒称v3的新证据。图像仅sklearn digits，不能声称自然图像泛化。
- 所有小批计划/runner已保存，紧凑证据在research/a_window_20261121；给B的接入说明docs/A_TO_B_NEXT_HANDOFF_20261121.md。下一步接B适配器的匿名完整验收，再重新冻结A2状态并按账本条件启动真实模型实验。

## A2-02：v3真实历史状态重建完成

- 运行前协议batches/A2_02_plan.md，复用scripts/run_a2_state_study.py prepare；生产源码提交1d57fd4，source_unchanged=true。
- 四任务12次真实fit、12冻结状态、97.100秒、真实API0；两组共同字段逐项相同。原始目录runs/a2_02_v3_states，脱敏JSON在research/a2_02_v3_state_preparation。
- 仅完成prepare，真实ask及动作质量评价仍未执行。11条历史未知用量仍在，ready_to_call=false。
- 加上前述73fit，本轮累计85真实分类器fit，其中流程/准备47、回归38；5模拟HTTP、真实API0。准备批未打破三小时计划49次实验fit上限，回归另登记。
- 首轮代码/结果已推送1d57fd4；续跑说明A_LINE_RESUME_20261121.md。后续文档/证据提交不修改生产源码，保持这12个状态的生产指纹。

## A2-03：封存决策的公平动作评价器

- 计划/结果：batches/A2_03_plan.md、batches/A2_03_results.md。tools/a2_action_evaluation.py及run_a2_action_evaluation.py支持validate-only，真实完整24提案与fixture严格区分。
- 两固定前缀的确定性动作小试：4真实fit、8逻辑评价、4复用，16.643秒、API0；同选结果一致，不同选两划分差值方向相反。是评价器验收，不是LLM反馈收益。
- 13项输入拒绝检查通过；最终preflight通过，原始目录runs/a2_03_evaluator_smoke和a2_03_input_checks，脱敏证据research/a2_03_evaluator。
- 本轮新增4fits，此前85累计89，真实API仍0。未修改superagent生产源码，A2-02状态仍保持原指纹。真实提案、动作质量正式结果与B调度自主验收仍待后续。

## A1-01：任务语义与实际训练一致性

- 状态：已完成 baseline、修复、fixed 与回归，2026-11-20。
- 计划：batches/A1_01_plan.md。
- 脚本：scripts/run_a1_semantics_batch.py。
- 目标产物：主仓库 runs/a1_01_baseline 与 runs/a1_01_fixed。
- API：0；runner 完整分类器拟合：0，只拦截训练入口。
- 结果：baseline 1/9（1.248 秒），fixed 9/9（1.218 秒）；四个隐藏翻转 recipe 在 symmetric 下均被拒绝，preference 兼容，歧义概率不确认。不是性能或独立测试集准确率证据。
- 回归：91 项通过，runner 墙钟 42.418 秒；tests.log 内测试执行 40.742 秒。包含小模型拟合，未逐项计数，不将它们计为性能实验的零拟合。首次手工回归误填了不存在的 tests.test_recipe_models，命令失败；已改为实际存在的测试清单，完整重跑通过，未重试 API。
- 修改：task_semantics、worker、benchmark、confirmation、finalization 及任务语义测试；执行边界同时覆盖注册 recipe、搜索和 refit。未实现真正 symmetric 算法，新候选应与 B 协同。
- 交接：batches/A1_01_results.md；research/a1_semantics_guard 保存可共享证据。原始运行目录含 a1_01_regression。当前工作区未提交，protocol 文件哈希标识实际运行源码。
- 后续：A1-02 核对配置对象与证据引用；A2 付费实验尚未开始。

## A1-02：配置事实与测量引用

- 状态：已完成 baseline、修复、最终 fixed 与冻结源码回归，2026-11-20。
- 计划：batches/A1_02_plan.md；脚本：scripts/run_a1_evidence_batch.py。
- 资源：真实 API 0、机制实验完整拟合 0；回归夹具另计。
- 修改范围：estimator_facts、measurement_claims、prose_claims 与相关测试。
- 结果：baseline 6/19（1.198 秒），最终 fixed 19/19（1.190 秒）；合法配置/测量仍可确认，错 trial/split/metric 不再借用别处数值。未知、失败试验、否定/未来与无作用域数字不伪装成已确认事实。
- 回归：123 项通过，测试执行 40.905 秒、runner 墙钟 42.614 秒，source_unchanged=true。首轮回归期间实现改变，已保留但不作为冻结证据，最终完整重跑通过；真实 API 调用始终为 0。
- 交接：batches/A1_02_results.md；research/a1_evidence_binding 保存 baseline、fixed、regression 和 runner 快照。原始最终目录为主仓库 runs/a1_02_fixed_release 与 runs/a1_02_regression_final。
- 限制：这是已知故障开发回归，不是独立泛化准确率，也没有真实性能提升证据；没有更改 LLM 输出契约或接入下一请求。
- 下一步：A1-03 核对结果进入下一请求；本批不宣称已完成自主纠错闭环。

## A1-03：执行核对进入下一请求

- 状态：baseline、fixed 与最终回归均完成。
- 计划：batches/A1_03_plan.md；脚本：scripts/run_a1_loop_batch.py。
- 本批用受控缺失元数据和确定性模拟客户端走真实 run_one，不证明真实 LLM 的纠错或性能收益。
- 结果：baseline 4/9，11.321 秒；fixed 9/9，11.120 秒；各 6 次真实拟合、4 次模拟调用，合计 12 拟合/8 模拟调用/真实 API 0。fixed 两次请求的共同上下文逐项相同，仅 E 增加 checks/memory。
- 回归：129 项通过，测试执行 40.543 秒，runner 墙钟 42.244 秒，source_unchanged=true。首轮新增六个测试夹具遗漏 MetricSpec 的 labels 参数而失败，已修复并完整重跑；未重试任何 API。
- 修改：contracts 保留 fitted estimator 与配置核对，checked_history_policy 显式投影派生检查，hypothesis_memory 按 trial 保留核对，benchmark 不再要求 assessment 才审计，explanation_audit 包含 hypothesis。
- 证据：batches/A1_03_results.md；research/a1_feedback_loop 归档协议、状态、实际提示、核对、结果与回归日志；原始目录 runs/a1_03_baseline、a1_03_fixed、a1_03_regression_final。
- 限制：下一候选由模拟客户端预登记，两组候选相同，反馈只改变其说明；没有证明 LLM 会换动作或提高准确率。本批无复核与最终 refit，不是完整交付性能运行。
- 后续：A1-04 正式 CLI 的策略/账本接入；之后再计划真实 A2。

## A1-04：正式 CLI 与费用/失败链

- 状态：baseline、实现、最终入口验收和宽回归均完成。
- 计划：batches/A1_04_plan.md；脚本：scripts/run_a1_cli_batch.py；入口说明：history_planner_cli_zh.md。
- 真实 API 均为 0。模拟 HTTP 沿用真实 ExperimentLLM 的 SSE 解析、请求上限和 SQLite 记账，模拟费用不计实际用量。
- 首轮 fixed 已完成训练与故障检查，剩一项配置测试写成 self-review 而实际枚举为 self_review；修正后新目录完整重跑，旧失败保留。
- 最终结果：5 项入口测试全部通过，29.218 秒，17真实拟合/5模拟HTTP；累计含前轮为34真实拟合/10模拟HTTP；真实API始终0。两模式完成三划分确认/refit/导出，模型重载各15条预测一致。
- 故障：timeout与缺usage均费用未知并阻止后续任务调用；无效JSON保留已知模拟token费用；均首请求后停止、无随机恢复，保留anchor模型并写stopped outcome。
- 回归：141项通过，43.281秒，source_unchanged=true；首次回归默认编码读取JSON在中文路径出错，修为UTF-8后完整重跑。回归夹具拟合/模拟调用另计未逐项汇总。
- 交接：batches/A1_04_results.md；research/a1_cli_entry归档协议、实际请求日志/输出摘要和回归。原始成功目录runs/a1_04_fixed_final与runs/a1_04_regression_final；模拟账本不等于真实账本，不导入真实费用。
- 下一步：完成记录后制定 A2 真实调用协议；B 的动态调度接入口与强方法线仍另行推进。

## A2-01：真实任务冻结状态（付费调用尚未启动）

- 已按运行前计划 batches/A2_01_plan.md 完成prepare，脚本scripts/run_a2_state_study.py。
- 四个真实任务各3次搜索，共12次真实拟合、12个状态，50.225秒、API 0；SMS按macro_f1，其余accuracy。训练外层留出及正式validation/test不进入状态。
- 输出主仓库runs/a2_01_states；源码指纹冻结、R/E共享字段一致、source_unchanged=true。
- 用户确认11条历史未知费用未对账；本机没有已核验真实累计账本。真实ask暂停，不能用模拟账本或空账本绕过。ask脚本和24请求上限已写好，待实际历史对账后preflight。
- 当前协调任务：我们A，队友B；按AB_INTEGRATION_V1.md提供共同Git基线和输入/动作样例，优先接通匿名小任务再做大实验。

## A/B 交接与发布

- 已写docs/AB_INTEGRATION_V1.md，明确共同候选、实际11fit/3请求、完整匿名验收、文件归属和仍待接入的Scheduler。
- tools/export_b_handoff.py从已执行run导出任务、候选/历史、带split/row/hash的逐行预测与预算；examples/ab_v1有匿名真实训练产物导出的输入，以及explore/confirm/stop格式样例（样例动作不是算法收益证据）。
- 新增证据完整性测试1项通过：保留预测行序、预算和refit预留，篡改预测会被拒绝。原A1最终回归141项及CLI5项仍为对应源码哈希的记录。
- A2本地prepare摘要/状态归档research/a2_state_preparation；真实调用依然0。准备将本轮A线与计划/证据发布到codex/expanded-feedback，精确提交号以Git交接消息为准。
