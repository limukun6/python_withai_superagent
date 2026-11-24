# A2-03：下一动作的配对评价器与离线验收

运行前协议，基线82f36dd。生产superagent源码不改，保持A2-02状态可用。只新增tools/scripts/tests。

问题：模型换了说法或候选，不代表下一步更好。先封存raw/checked决策，再在相同两份未展示给决策器的训练内部划分评价。固定seed2707/3707，继承A2-02的reserve41101和outer45101；不评分外层、不读取原包validation/test。

正式模式：只接收A2收集器完整24条决策，核对prepared协议hash、生产hash、状态hash、上下文hash、任务/指标/候选/参数和原始提案一致性；任一错误在训练前停止。缺失/失败决策不被静默忽略或补成有效决定。
每个不同(task,candidate,split)只拟合一次。相同动作跨状态/组共享测量，明确复用，不把共享结果算独立训练样本。最坏12状态×2动作×2划分=48真实fit；只允许当前expanded静态单分类器能力及完全相同参数。后续候选库修改时先审计成本和新协议。
结果按task/prefix列E−R主指标差、每划分差、同选率、胜平负；两个划分平均仅为诊断估计，不做显著性检验，不把前缀视为独立任务。失败记录真实次数和未知component成本，停止本批，无自动重试。

本次只跑fixture-smoke：从DryBean两个固定前缀的共同可用目录按原顺序取前两个动作；一状态两组同选、另一状态不同，用确定性动作代替模型。预先固定，不看分数挑动作。预期4次唯一训练、8次逻辑评价，重复复用4次；API0。结果只证明评价器与复用机制跑通，不作为真实LLM收益。
另做无训练输入验收：重复组、缺组、参数篡改、状态篡改、指标/上下文身份、伪造完整性、超预算在训练前拒绝。先保存runner再执行。

执行器scripts/run_a2_action_evaluation.py；smoke脚本scripts/run_a2_evaluation_smoke.py。输出新目录protocol/checkpoint/results/completion；先validate-only核对计划实际fits与动作不执行训练。真实API仍因历史11条未知用量未核验而保持0。

小试后补7项零拟合输入验收，scripts/run_a2_evaluation_input_checks.py：冻结状态字节、context指纹、未登记seed、依赖版本、决策批绑定、未完成调用批、非bool组标识。新增依赖/seed检查和评价器源码指纹只影响preflight/日志，不改变已测动作或分数；最终版本validate-only再次通过，不无故重跑4个训练。
