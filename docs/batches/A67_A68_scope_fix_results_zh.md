# A67/A68：先核对比较对象，再判断假设是否支持

## 瓶颈与结构

旧overall/slice假设检查在明确跨主指标、split或evaluation payload的条件下，仍可能只因数值上涨标supported。逐条测量引用已有绑定，不代表“两条测量相减”也绑定了同一作用域。

新增comparison_scope共用检查：显式不一致或单侧身份字段不全则not_evaluable；比较primary score时检查primary_metric/metric_spec；切片另查定义与可用行指纹，重复trial/slice不借用第一个记录。正式AB每个观测增加从真实worker prediction_record来的payload/label/rowID指纹，原Flash公共上下文不发送标签和整个预测向量。

旧入口两边均缺身份时保留旧数字判定，但报告provenance_complete=false；本修复不是所有旧路径都已完成身份审计，也不证明自由文本因果正确。None或非SHA值不冒充完整provenance。

## 实验

A67在执行前保存8类48个A侧开发案例及标签依据。当前旧源基线44/48，slice_scope与failure_comparison各2个错误确认；修复后48/48，合法案例未因一律拒绝被掩盖。此开发集用于修复，原B侧独立48仍未到位，不能宣称原96/48封存验收已完成。

首次runner缺memory fixture的search_accuracy而中断；修fixture后另目录完整基线。首次生产修复将slice scope代码放错函数，NameError中断；原失败目录保留。未改案例预期、未读H/test、无API重试。

A68一条匿名numeric正式CLI（local/ab-v1/3fit）完整执行，2条观测都有真实测量身份；1个合法比较可评价，改split/payload/label/primary metric或删单侧行指纹的5个受控反例全not_evaluable。模型新进程predict通过。3实际fit/9.624秒/API0。之后仅完善provenance_complete元数据的空值/格式说明与判定，A68保留其原源SHA，不把旧运行写成最后元数据修订后的实训。最后48例与52项相关无训练测试通过。

## 资源更正

55项阶段回归包含`tests.test_hypothesis_agent.IntegrationTests`，其mock worker仍调用真实Dummy.fit。按当前未改源码和成功记录回溯：双臂6 + 故障恢复3 + one-shot3 + 导出3 = 15监督fit。先前“无训练检查”判断有误，补记15；这是回溯计数，非当时物理观察。测试用模拟客户端，真实API为0。其它相关测试不额外训练。

A66后1007 + A67回归15 + A68物理3 = **1025**。后续52项检查只选择HypothesisTests并排除IntegrationTests，无新增fit。总真实API0；不能把48案例执行零训练误推为整个修复过程零训练。

后续A69 B1矩阵与A70四任务固定状态已独立写计划/有限预算/runner，运行状态看各自completion。它们不回填成本到本批。
