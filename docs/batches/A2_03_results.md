# A2-03：公平评价下一动作的工具已验收

问题：Agent改变候选或者解释不能直接算效果改善；相同动作重复训练还会浪费预算、夸大样本数。

结构：新增tools/a2_action_evaluation.py及三个脚本。先封存决策，核对prepared/源码/依赖/状态/上下文/提案/参数身份，再在固定2707/3707两份共同训练内部划分执行。每个task/candidate/seed只训练一次，并检查预测指纹与重算主指标。正式输入需完整24条配对提案，失败停止，不自动重试。superagent生产源码未改，A2-02的v3状态仍可用。

实验：两个DryBean前缀、按目录顺序预定两个候选。第一个前缀raw/checked同选，第二个不同选。两组名称仅接口fixture，不代表真实LLM产生决策。

- 8次逻辑评价、4次唯一真实fit、4次复用；16.643秒；真实API0；外层未评价。
- 同选状态两划分差值均0。不同选状态两划分差约−0.001435、+0.002869，平均+0.000717。这仅展示评价器输出与划分敏感性，不是反馈机制增益或统计显著性。
- 6项初始拒绝检查、7项补充拒绝检查共13项通过：重复/缺组、未知组、不可用动作、改参数、fixture冒充真实，以及状态/context/seed/依赖/批绑定/未完成批/非bool组。
- 补充检查后最终代码validate-only通过；新增的是依赖/seed校验和评价器日志指纹，训练评价算法未变，无新增重复训练。

本轮只新增4次分类器fit/API0；此前85次累计更新89次。该批另有运行前协议，不属于此前三小时计划的49fit额度。

本机原始证据：runs/a2_03_evaluator_smoke、runs/a2_03_input_checks。共享证据：research/a2_03_evaluator。生产状态源码仍1d57fd4；评价器代码指纹另行记录。

## 正式接跑

在integration工作树、真实24条提案已按既有ask协议完成后：

```powershell
D:\codex\python_withai\.venv\Scripts\python.exe scripts/run_a2_action_evaluation.py --prepared D:\codex\python_withai\runs\a2_02_v3_states --calls REAL_CALLS_DIR --output NEW_VALIDATION_DIR --validate-only
D:\codex\python_withai\.venv\Scripts\python.exe scripts/run_a2_action_evaluation.py --prepared D:\codex\python_withai\runs\a2_02_v3_states --calls REAL_CALLS_DIR --output NEW_EVALUATION_DIR
```

REAL_CALLS_DIR和NEW_*需替换为实际路径，新目录不能存在。正式模式不要传--fixture。正式评价最多48唯一fit/API0；真实模型调用依旧待累计用量核验，此批未触发ask。

解释边界：只覆盖当前冻结expanded单分类器目录和现有四任务。B整合新增方法或多组件成本后必须更新能力/预算协议；动作质量实验不同于完整Agent最终任务成绩。代码共享已准备，真正Flash行为和质量结论仍待真实决策。
