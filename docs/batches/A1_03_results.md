# A1-03：执行核对已进入下一轮请求

本批在计划与 runner 落盘后，使用真实 run_one 搜索循环、真实训练 worker 和确定性模拟客户端完成 baseline/fixed。不是正式任务性能实验，也没有真实 LLM 调用。

## 瓶颈与结构

原流程会记录配置矛盾，但 DecisionState 的字段白名单将其丢弃，导致下一请求缺少事实。原解释审计还依赖 assessment，漏掉了 CheckedHistoryPolicy 的常见提案及 hypothesis 中的配置/测量陈述。

本次修改：

1. ExperimentResult 和 DecisionState 保留 worker 的 estimator 与 configuration_fact_check。原始状态只传数据对象信息，不传模型对象本身。
2. 两个提示组共享实际 estimator 和原始观测；检查组的 checks 增加配置核对、有限范围解释警告。两组 history 都屏蔽派生核对；raw 不获得 checks/memory。
3. memory 按 trial/candidate 保存配置核对与解释警告，后续确认不会覆盖旧矛盾。性能假说检查依然只说明测量是否支持，不升级成因果知识。
4. 所有非空提案都审计，包含 hypothesis；未来预测不能因为审计入口变广而成为已确认实测事实。

## 做了什么实验

构造一个固定数值二分类任务。仅对指定 ExtraTrees 候选的事前元数据注入 unknown，模拟新工具缺少说明；模拟提案误称 logistic，而实际 worker 仍真实拟合 ExtraTrees。另在 hypothesis 故意声称 trial1 搜索达到 99.99%。没有改变 worker 返回的真实算法身份或成绩。

raw/checked 各三次搜索，无额外确认、无最终 refit，各两个模拟请求。两轮各 6 次真实分类器拟合，合计 12 次。模拟客户端看到配置矛盾后改变其下一条说明，但两组下一候选仍是预登记的同一候选；不把这种程序分支当真实推理能力或选模收益。

| 验收 | 修复前 | 修复后 |
|---|---|---|
| worker 记录实际对象与声明矛盾 | 有 | 有 |
| 下一状态保留实际 estimator | 无 | 有 |
| 不含 assessment 的提案被审计 | 无 | 有 |
| hypothesis 的错误测量被审计 | 无 | 有 |
| 下一检查组请求获得配置矛盾 | 无 | 有 |
| 检查组获得假说未通过 | 有 | 有 |
| raw 不泄漏派生检查 | 通过 | 通过 |
| 模拟客户端读到并回应冲突 | 无 | 有 |
| 每组 3 搜索、2 模拟请求、零随机恢复 | 通过 | 通过 |
| 汇总 | 4/9 | 9/9 |

baseline 墙钟 11.321 秒，fixed 11.120 秒，均 source_unchanged=true。总真实 API 0、课程 API 费用 0、模拟调用 8、性能控制器探针真实分类器拟合 12。另行回归的模型夹具未逐项计数，不混入这 12 次。

六个新增接口测试覆盖：实际观测可达、R/E 仅派生字段不同、旧矛盾保留、隐藏反馈不泄漏、hypothesis 审计、投影不修改历史。最终回归条数、耗时与 source_unchanged 见归档 completion/tests.log。

最终回归 129 项通过，测试执行 40.543 秒，runner 墙钟 42.244 秒，source_unchanged=true。首轮六个新增测试因夹具缺少 MetricSpec.labels 而失败；修复后完整重跑，失败日志保留。fixed 实际两次请求逐项比较：去掉 E 的 checks/memory 后与 R 相同，结果保存为 context_comparison.json。

## 证据与边界

原始输出位于主仓库 runs/a1_03_baseline、a1_03_fixed、a1_03_regression。可共享协议、逐项结果、提示快照、trials/states 与回归日志归档在 research/a1_feedback_loop，不共享训练模型或临时数据缓存。runner 与核心源码身份均有哈希。

这是人为故障的确定性集成实验，不估计 LLM 自然犯错率，不证明反馈带来真实选择改善。没有完成最终 CLI 的全部策略开关与账户挂接；模型本批也未完整 refit，不能当作正式交付模型。

下一批 A1-04 先写计划与 runner，接正式入口的 raw/checked 开关、累计费用账本和失败停止，使用模拟客户端完成入口验收；之后才能按 A2 协议开展同历史真实调用。B 的强方法与调度线仍需独立推进。
