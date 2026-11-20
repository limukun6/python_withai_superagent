# A-06 执行协议：指标与正类沿真实控制循环传播

四指标accuracy/macro_f1/binary_f1/roc_auc，每个合法/冲突/缺证据三条，合计12。
合法轨迹使用固定预测器替代训练，真实run_one、DecisionState、结果契约、confirmation、finalization；每个执行器调用声明classifier_fits=0。不是实际分类器训练或性能结果。
两个固定预测器在不平衡标签上让accuracy与F1排名相反。AUC提供类序对应分数。冲突轨迹显式改metric_spec名称或正类顺序；缺证据轨迹不提供完整metric_spec，要求结果记录unknown，不把旧格式补猜成已确认。
先baseline后fixed，不因观察结果改期望。所有调用本地，0真实训练/0API。预期合法4条不退化、冲突4条拒绝、缺证据4条unknown。
runner scripts/run_a06_objective_flow_batch.py --source-root SOURCE --output NEW_DIR [--validate-only]。
baseline源码0b112d8；fixed只补实际目标身份，不改变算法或评分公式。每条记录各阶段及选型；错误保留，不自动重跑。

后续投影核对：新增metric_identity为派生检查，只出现在checked组checks；实际metric_spec两组共享。TaskSpec目标来源两组共享，instruction同样提醒默认不等于已确认意图。context_version升级history-evidence-v3，旧A2冻结状态不可冒称新版本。运行tests.test_feedback_loop六项（0 fit/API）验证去掉checks/memory后两组上下文逐项相同。
