# A151堵点二复核：先压缩候选表示，原故障归因需降级

2026-11-24，基线6ecaaa8。已完成免费复核与可逆编码原型，未发学校API、未训练、未修改生产模块或真实账本。此文是对A151_blocker2_for_codex的回复；不是宣称原闭环已修复。

## 1. 先纠正三个判断

1. **可以确认两次短预算重放出现推理耗尽，不能确认原始16384预算调用也是如此。** Flash的1024和Pro的2048重放均finish=length、无content；原调用是RemoteDisconnected、无最终usage。它可能长推理、排队、网关保活/转发/上游断开，缺服务端trace不能唯一定位。更不能称这个payload“不可解”。输入大、字段多和推理时间长目前是待检验关系，不是已证因果。
2. **本机当前标准HTTP栈不会偷偷重发约七次POST。** 实际环境requests 2.34.2 / urllib3 2.7.0，Session默认Retry(total=0, read=False)。对连接、读取、协议错误分别调用实际retry策略均立即抛出，不允许重试。ExperimentLLM自己新建Session、无重试循环、禁止redirect；campus_direct只改进程内域名解析。不能因417≈7×60就推断客户端重复调用。网关内部重试则客户端无法免费查清，需服务端请求trace；这也不是能把unknown结算成零的依据。
3. **原故障发生在L0 calibrated-v2，没开启grounded。** 失败payload实际instruction是11个字段的calibrated契约，不含grounding输出包；L1未启动。不能据此说“grounded的4子字段导致失败”，或说“分层机制真实闭环失败”。失败点在两条路径共享的规划调用层。

## 2. 为什么60秒能等417秒，为什么新计数仍不够

requests的read timeout通常约束等待读取数据的间隔，不是完整请求墙钟上限。持续字节可以让它继续等，DNS/网关/操作系统等也不能简单用整数倍推断。

本机Response.iter_lines默认chunk_size=512，会经iter_content缓冲后才分行。若不足一块的数据在异常前没被交给上层，即便TCP收到字节，事件计数也可能为零。新代码还用`if line:`排除了空行：只收到空行时other_events仍为0。离线夹具重现了这一盲点。

因此：other_events>0可证明看到了非data行；other_events=0**不能证明**连接没有字节，更不能推导“必然HTTP重试”。data_events=0也不等于reasoning_tokens=0。原失败无用量继续保持unknown。

后续如果修改传输观测，建议显式零重试策略并记录版本/策略、空行/注释/data分项、原始接收字节及首末字节时间、可用的服务端request-id、终止类型。字节观测需在iter_content层，不能只看分行后的记录；观测仍不保证打破阻塞读。严格墙钟需要独立取消/进程控制，并将被中断请求保留unknown，不能宣称for循环deadline能随时中止。不要为收集失败再付费重放原请求。

## 3. 具体解法：菜单表示压缩优先，暂不拆成两次调用

先保持同一候选全集、参数含义、history/checks/memory与输出核验要求，把候选表的重复列名及重复配置共享。只有表示变，代码解码后必须与原payload相等。原始冗余既在菜单又在candidate_facts，先去掉占比仅约10%的检查信息不是主要杠杆。

已保存离线原型`scripts/review_a151_blocker2.py`，没有接生产CLI：

| 原型 | 实际结果 | 用途 |
|---|---|---|
| 逐节点带标签表格 | 整体字节反增2.356% | 负结果，保留v2目录 |
| 仅共享重复值 | 整体字节减6.625% | 中间结果，保留v3目录 |
| 共享列名表＋重复值 | 20030→12919 UTF-8字节，减35.502% | 当前原型，完整payload可逆相等；29个候选全部保留 |

字节口径统一为原/新紧凑JSON，包含新编码说明，避免把去空格算成结构收益。**不是token实测，不能声称少35.5%推理或省35.5%费用。** 模型读取表和引用是否更容易也尚未知。现有9项定向测试通过，包括编码还原、缺失/null、保留关键字转义、旧字段不变、客户端零重试与空行盲点。各次离线运行耗时见completion，不把程序毫秒当人工研发时间。

不建议先把一次拆成两次：重复输入、额外时延、占用每路径3次额度，且第二次还可能失败。只有明确阶段边界且共享预算可核对，才能另立方案测它。

若仍需输出瘦身，应新增明确版本的协议。例如只选注册candidate_id、purpose、短reason和必要evidence_refs，固定task/配置由本地目录带出并标注“系统派生”；不能伪装成LLM自己核实的断言。grounded需要的引用/问题处理不能暗填，瘦身协议不能冒称原grounded对照。直接换成candidate/params/reason还会允许模型改注册参数，增加不一致，推荐选ID后从目录取精确参数。

## 4. 推荐执行顺序及停止条件

1. **先接表示开关**：如显式menu-view，旧默认不变，原始状态/目录留本地，提案仍按原capabilities严格验证。编码不得影响B排序、eligibility、fit预算与选型。grounded事实身份仍绑定原目录，不对压缩表示重新发明事实身份。
2. 离线测试至少覆盖本地多类任务、所有候选/参数保留、序列化往返、旧默认上下文不变及真实状态回放。当前只实测A151一个失败状态，不能叫全任务无损验收。
3. 另写单调用机制计划/runner，先validate。**最多1次新Flash、0fit**，用原已有状态的压缩表示，原calibrated-v2输出契约，不同时改模型/输出协议/预算参数。请求计入共享真实账本及对应研究请求上限，原失败不续跑不重试。不是重复原始payload；须记录表示变更及原SHA。成功也只证明一次合法提案可返回。
4. 首次机制成功后才另定小规模端到端验收，仍每路径最多3次调用/登记fit上限。若机制失败立即停止，不自动尝试Qwen。Qwen可作为另一个合规研究组，但需单独接preflight/served-model/计价和输出协议，并非当前只认Flash的入口换个model就能跑。
5. 若压缩表示仍不可靠，先保留本地正式路径。Flash/grounded注明实验性开关和适用边界。不要为挽救结论连续换参数、加token、关thinking再重试，已有诊断不能当成没做过。

费用阈值：保持现有本次请求上限及既有预留机制，不能用原输入估计/一次折算冒称学校最终扣费。本批没有新请求；第3步尚未实施或执行，不代表已经获新实验结果。

## 5. 本地正式路径对报告的影响

本地规划仍有A的任务/指标契约、训练内划分、执行与预算、历史/复核及模型交付。通用性取决于同一冻结系统能否在不同真实任务自主运行，并不要求每一步都调用LLM。已有H1、Banknote等按原来源保留。

但不能因此宣布课程对“LLM自主开发”的要求已经满足。L0能否作为最终提交路径，要对照课程具体要求；若要求运行时必须调用Qwen/DeepSeek，应完成一个合规LLM路径再提交，否则明确未满足项。正式报告应把结构可靠性、任务覆盖、LLM选择收益分开，不能用本地性能替代grounded真实闭环验证。

## 6. 当前账本与复跑交接

当前按最新导入记录：396请求、known课程折算¥7.130568、unknown16、pending0；快照`research/a130_accepted_history_v3/accepted_history.json`。旧CURRENT/PROGRESS顶部388/13已过期，需要更新。这16条包括12历史、A151失败1条、无课程计价GPT诊断3条；GPT只能诊断，不能放入正式模型选择结果。接受清单不代表这些金额被核清，新增unknown仍停。

本次0fit/0模型API/¥0新增；A累计按现记录仍1651fit。本机标准HTTP策略的复核不消除原未知，也未修改账本或owner。

复跑离线脚本（只读旧events，新输出目录）：

```powershell
Set-Location -LiteralPath 'D:\codex\python_withai\runs\integration'
$py = 'D:\codex\python_withai\runs\a79_encoder_assets\venv\Scripts\python.exe'
& $py -m unittest tests.test_a151_blocker_review tests.test_reasoning_switch -q
& $py -m scripts.review_a151_blocker2 --events 'D:\codex\python_withai\runs\a151_small_trial_attempt_02\L0\run\events.jsonl' --output research/a151_blocker2_review_next
```

结果：`research/a151_blocker2_codex_review_v4/{protocol,results,completion}.json`。首次脚本将event键误写为event而实际为kind，0成本拒绝，保留`research/a151_blocker2_codex_review/`；修正后才得到各原型结果。生产`superagent/`未改，菜单开关/传输字节观测/下一次真实探测均未实现。
