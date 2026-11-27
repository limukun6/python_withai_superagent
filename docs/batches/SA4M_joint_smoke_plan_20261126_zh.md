# SA4M：合并版真实worker与输出交付烟测

在SA4 150项无fit回归通过后登记。只验证合并后的工程链，不评价真实LLM语义能力或真实任务迁移。本批同一新生成的120行标量分类小任务，两条正式统一train入口：B默认fallback；A proactive＋B opening-local＋protected self-review。同一完整候选目录，不手选候选或修改B排名，seed18643，每条fit_limit=2（含重训）、wall120秒、worker沿CLI默认timeout300秒并受剩余wall约束。两条合计最多4次物理fit、真实API上限0、课程API费0。

主动解释使用脚本化JSON客户端：1条语义回复，非真实DeepSeek。输出要求保留6位原ID、Y/N映射、BOM分号CSV、按样例逆序。预测包与训练包分开。通过官方train函数→CLI handler→真实_train→真实worker选择/训练/重训；每条模型交给真实新进程predict。默认组保留旧CSV格式，组合组按公告输出，反映格式约束增量。按ID恢复顺序后核对两条预测，不以相同预测作为性能提升。

runner先落盘再执行。私密任务包、worker负载、模型、预测放在忽略的runs/sa4m_joint_smoke_NEW；脱敏protocol/results/completion放research/sa4m_joint_smoke。保留失败回执，不以创建新目录自动重复训练。失败或超预算即停止，不运行课程Task1/Task2、不查看正式测试、不写真实账本。
