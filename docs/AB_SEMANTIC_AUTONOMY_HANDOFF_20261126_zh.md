# A语义适配与B自主版已合并：接手说明

2026-11-26。用户要求将A刚完成的SA1–SA3合入B最新自主版。共同研发分支为 **codex/ab-semantic-autonomy-1126**，从这个分支的仓库根目录运行。它是阶段性集成，继续开发，尚未做最终冻结。

## 用的是队友哪一版

- B完整研发来源：codex/task-driven-planning-1125，**b0944ae**。
- B个人分支发布：**c6ea0b7**的releases/autonomous-integration-2026-11-26。127个模块的Git对象及unified_agent.py与b0944ae根目录逐项相同。无需把个人分支的旧根目录误作最新版。
- A来源：**0d09760**，含SA1全局语义契约、SA2输出recipe、SA3主动解释/字段工具/训练前试读。
- 双方共同祖先d501807；用真实Git merge保留双方历史。三处冲突是autonomous_run.py、superagent/unified.py、unified_agent.py，均为逐段组合。保留B的开题、反馈、方法目录、规则继承和恢复交付，A在其前后接入读题/试读/输出；没有改B算法排名。

旧Task1 1533/2030和Task2 684/940属于B已验证版本，**本轮未重评课程验证，不能写成本合并版新实测成绩**。

## 怎么用

未启用语义模式时仍使用fallback，B的原有控制选项保留：

```powershell
python -m unified_agent solve TASK.zip --prediction-input FEATURES.zip --run-dir runs/NEW --output runs/NEW/predictions.csv
```

已配置真实校园client后，主动语义适配和B开题可以组合：

```powershell
python -m unified_agent solve TASK.zip --prediction-input FEATURES.zip --run-dir runs/NEW --output runs/NEW/predictions.csv --task-interpretation proactive --task-analysis flash --protected-self-review --fit-limit 10
```

proactive会消费真实模型额度，需既有密钥/账本/任务累计ID配置；不是开关存在就代表校园连接和历史并账已验收。`--task-analysis local`仅表示B诊断不付费，若同时显式proactive，A读题仍需要模型client。未配置时不能把该路径算成成功的LLM组。不要把本轮脚本化client当上线配置。

统一共享调用计数：原B开题路径保留最多1次；显式proactive＋开题最多使用原client允许的3次，语义工具、提案和修复先计入，开题只在有剩余额度时调用一次，达到上限就记明原因并继续已有本地合法路径。显式0或更小上限不扩大。研究outbox原先限制1次的client也不会被扩为3次。网络失败不重试。

B研究能力允许显式16fit仍保留；团队公平对照每路径11fit及本轮10fit默认不因此改变。

## 已经验证什么

| 验证 | 结果 | 资源与边界 |
|---|---|---|
| A原100项＋B相关43项＋7项组合检查 | **150项通过**，10.500秒 | 0fit、0真实API；B两条直接调用DummyClassifier.fit的旧夹具明确排除，不算通过 |
| 合并默认路径 | 自主选RandomForest，搜索1次＋全量重训1次；独立CLI输出20行 | 合成120行训练任务，2次真实fit |
| A proactive＋B opening-local＋protected self-review | 同样自主选RandomForest并完整重训，B开题/自审记录存在；独立CLI输出20行 | 同任务，2次真实fit＋1次脚本化语义回复，0真实API |
| 自定义提交 | 6位ID、Y/N映射、UTF-8 BOM、分号分隔、按模板逆序全部正确 | 对齐ID后两路径预测值完全相同；不声称提分 |
| 来源保留 | A六个核心模块及B十三个模块与各自来源Git对象相同 | 共享入口/循环和TaskOpening预算桥为明确合并改动 |

两条烟测共**4次真实分类器拟合、27.204秒、真实API 0、课程新增费用¥0**。不是0训练；也不是未知真实任务或真实LLM成功率实验。两条烟测在Python3.12.14本机完成，B历史分数的Python3.11环境保持其原身份。

首轮150检查有4错误：两处Windows SQLite测试连接未关闭、旧mock需随读题入口迁到semantic_contract、组合client夹具缺spent字段。修复后150通过；旧mock仅调整注入位置，断言保留。烟测runner第一次在导入前有多行with语法错误，0fit/API；修正、py_compile后实际烟测通过。全部失败回执保留。

## 文件、复跑和下一步

- 无fit计划：docs/batches/SA4_AB_merge_plan_20261126_zh.md；runner scripts/run_sa4_ab_merge.py；最终证据research/sa4_ab_merge_v2。
- 真实烟测计划：docs/batches/SA4M_joint_smoke_plan_20261126_zh.md；runner scripts/run_sa4m_joint_smoke.py；证据research/sa4m_joint_smoke_01。
- 来源/物理fit审计：scripts/audit_sa4_merge.py，证据research/sa4_merge_audit。模型、任务包、预测在忽略目录runs/sa4m_joint_smoke_01，不进Git。

```powershell
# 输出目录必须全新；第一个不训练，第二个最多4fit，不调用真实API
python -m scripts.run_sa4_ab_merge --output research/sa4_replay_NEW
python -m scripts.run_sa4m_joint_smoke --private-output runs/sa4m_replay_NEW --evidence-output research/sa4m_replay_NEW
```

接下来双方都从共同集成分支继续：A固定独立语义案例/oracle并验证真实Flash和三模态陌生输入；B继续方法目录和自主实验。已明确的边界保留：无manifest图像/数组的全局语义适配未补齐、冻结后才出现的预测schema还没有独立LLM修复入口、任意join/插件生成未实现。本轮没有完成四天开发计划，不要为补齐材料重复跑课程已见验证或把历史成绩搬到新源。
