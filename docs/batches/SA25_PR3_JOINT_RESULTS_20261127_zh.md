# SA25–SA25C：PR3与A最新结构整合结果

整合起点A=7e9d142066be7f72fa124a0dde37b068af1ef37e，B/PR3=4d4222aeb9f59f30b1596eca6e02709bb15a8072；共同祖先SA21=5f820bc。独立工作区runs/pr3_joint_1127，分支codex/a-pr3-joint-1127。本次本地整合保留双方历史；没有替用户在GitHub合并PR #3，也没有重跑Task1/Task2/H2分数。

## 瓶颈 → 结构

1. **两套交付流程互相覆盖。** B的first_delivery/final_prediction与A的预测恢复原先改同一个CLI。现在普通预测和分阶段交付共用selected_prediction：先用选中模型预测→查询已验证适配器→需要时走原恢复控制器→核验真实输入ID来源→发布后写独立凭据。输入可恢复时优先继续原模型；真正无法完成才使用已明确标partial的先验。发布本身出错不能再降级覆盖已验证结果。显式错误适配器不被静默忽略。
2. **组合后缓存和计费规则缺一边。** 合并26个reader源绑定，保留B限制/评分绑定和A native-v3设计记录、digest与恢复角色；显式解释迁移也执行相同bind_semantic_rules。保留B的短抽取/反馈输出限制以及A prediction_recovery=4096、剩余不足30秒不发请求。显式local/control在auto下不新建付费client，proactive组合保留。
3. **合法输入描述被当成算法禁令。** “utterance is the only input feature”先前触发RestrictionError。把输入特征角色与算法实体区分，保留真正未支持的算法禁令和约束检查；不是忽略未知限制。
4. **Windows资源和进程边界未接通。** 两处SQLite事务结束后未关闭连接导致Windows文件锁，现显式关闭。EOF监控线程和训练worker此前共用stdin；worker改DEVNULL，避免继承控制管道。控制器退出/超时只清理自身训练进程树，POSIX原逻辑保留。
5. **早期先验漏用已经接受的列映射。** 先验计数现在使用冻结field_map找到真实预测输入列；不填缺失特征、不改目标。B的标准family ID和模型导出修复完整保留。

## 做了什么实验

最终396/396相关无fit/API检查通过，36.078秒。覆盖A原360项、B限制/专门模型/反馈/资产与新增组合路径。明确排除两个原DummyClassifier.fit夹具和B有真实训练的formal_solve夹具；不称全仓所有检查。第一次393项有1失败/6错误，包含SQLite锁、规则误识别、本地模式配置和测试默认GBK读取UTF-8 JSON；原checks_01保留。checks_02–05分别记录逐步修复，不将失败删去。

正式默认CLI，stdin关闭，使用既有公开开发来源。普通任务manifest明确要求线性、禁预训练/外部数据；自定义任务显式迁移原SA21真实解释，预算内自主训练；晚到输入显式再验证原SA23真实LLM适配结果。没有向新模型请求回复，没有人工指定选中的分类器。

| 路径/证据 | 结果 | 已知实际拟合 | 新校园API | 机器秒 |
|---|---|---:|---:|---:|
| SA25B ordinary：Iris，96训练/24预测 | 早期partial→最终completed；新进程相同，执行方法符合三条限制 | 2 | 0 | 14.062 |
| SA25B custom：Breast Cancer，240训练/40预测 | 早期/最终均正确保留原ID、BOM、tab、指定B类概率；新进程相同 | 2 | 0 | 21.985 |
| SA25B late_replay：原40行晚到CSV | 实际原解码失败；显式再验证旧真实回复后两进程自动命中缓存，特征/CSV与参考相同，模型/规格/拟合记录不变，2份独立交付凭据 | 0 | 0 | 6.140 |
| SA25C Windows控制器丢失：原24行 | 4.938秒已交先验；在worker模块执行前暂停后结束真实控制器，6个launcher/解释器全退出；24行原文件不变、新进程一致 | 0 | 0 | 7.235 |

前三条来自SA25B同轮，合计42.515秒的该轮第四条夹具失败也保留。Windows来自独立SA25C补测，不能把SA25B原completion改写为一次4/4成功。

独立原始文件审计4/4，0新增fit/API：核验真实模型、早期/最终摘要、原ID/概率、冻结目标和特征，全部526条真实学校账本行不变。通过正式export_model_bundle将两个最终模型及对应源码导出到新目录，独立python -I入口禁网络和训练后预测；共64行分别与首次交付逐字节一致。此核验由A另一路代码完成，不是人类盲标。

## 失败和资源完整保留

- SA25 delivery_01普通路径152.844秒仅partial，worker超时：账本记录0次已知拟合、unknown_failed_attempts=1。没有worker fit回执，不把它说成确定0fit或完成任务。随后最小零fit管道探针：继承stdin超时，DEVNULL在0.328秒正常完成。probe_01的布尔标记误命中异常命令文本，probe_02按完整输出行修正，原日志不改。
- SA25 delivery_02普通路径完整2fit；custom最终模型/预测完成2fit，但首份先验因漏field_map失败。两条合计33.938秒。缺早期文件的整体验收失败，不能称这轮2/2。
- SA25B是单独登记的额外4fit/0API续批，不伪称仍在原6fit批次内。前三条成功；第四条观察器把Popen类替换为函数，asyncio继承时退出，未进入训练。这是夹具问题，不算Agent拟合失败；没有重训已成功模型。
- SA25C保留Popen类继承，只观察已创建进程的身份；任何夹具异常使对应测试进程退出，防止误入fit。单独0fit/0API补Windows验收。夹具不修改模型、标签、选择；仅在模块运行前暂停。不是实际关机、fit内部中断或磁盘损坏实验。

本次P0累计**8次已确认分类器拟合＋1次旧未计量worker尝试，新增校园API0、课程费用0**。形式路径合计236.532秒（含失败），另有零fit检查/探针/导出审计；不把已知fit数当成确定总fit数。真实A canonical维持526请求、已知课程¥8.428396、17unknown、0pending、所有行SHA不改。B待导入原账本缺口仍单列，不能称这是完整团队账本；原未知token/费用仍null。

## 支持到哪一步与下一步

证据支持此版本在本机Windows将早期交付、最终交付、已有解释/适配复用及模型导出接通。普通/自定义任务是已使用公开来源，manifest或显式复用的解释不是新LLM从零理解任务；晚到适配是旧真实回复的再验证，新的恢复机制由模拟client组合检查和既有SA23真实实验支持。本批没有新Flash恢复样本、质量评分、独立来源迁移，也不证明任意任务可完成。

P0整合完成，开发未冻结。下一优先P1：将可恢复的任务声明/限制/评分解析失败交给LLM查原文和修订受支持解释；继续保留真正未知目标和不支持能力的明确状态。之后补图像ID/原生-记录适配、不同原包的三模态无人干预验收。复跑脚本/预算先保存，不从重跑已有Task1/Task2分数开始。

计划：SA25_PR3_JOINT_PLAN、SA25B_JOINT_CONTINUATION_PLAN、SA25C_WINDOWS_OBSERVER_PLAN。凭据：research/sa25_joint_checks_01–05、sa25_windows_pipe_probe_01–02、sa25_joint_delivery_01–02、sa25b_joint_delivery_01、sa25c_windows_interruption_01、sa25_joint_audit_01及各prepared。脚本run_sa25_joint_checks.py、run_sa25_joint_delivery.py、diagnose_sa25_windows_pipe.py、run_sa25c_windows_interruption.py、audit_sa25_joint.py。私有包/模型/源码导出包不入Git。
