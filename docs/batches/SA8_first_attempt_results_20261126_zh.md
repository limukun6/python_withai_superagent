# SA8首轮：文本完整交付，两例被评分器误拒，原记录保留

本批计划SA8_real_delivery_plan_20261126_zh.md，统一入口结合A proactive与B opening-local/protected self-review，每路径最多4fit，共最多12fit；三种真实公开数据，独立预测答案只在本轮终局封存后打开。原始结果research/sa8_live_01，事前源码/旧账/计量审计research/sa8_audit。

首轮实际：V1数值0fit/2API、V2文本4fit/1API、V3图像0fit/2API。文本路径自主选RandomForest、完整重训、新进程生成60行原ID及标签映射CSV，答对55/60、准确率91.67%、macro-F1 0.699700。训练池为项目已用SMS训练数据，240/60新开发划分，不是新迁移性能。

数值和图像两例均已得到程序接受的真实输出overlay，并通过输入、指标、目标、类别、训练行及预测指纹核对，但实验oracle因output_recipe字典不完全一样而在standby之前中止。具体差别是模型显式写了恒等标签映射（例如0→0、1→1），预写oracle省略mapping。这两种方式生成相同CSV，**是实验评分器过严，不是Agent的目标解释或输出错误**。首轮的failed状态和accepted_wrong_contract原字段保持原样，另作测量缺陷说明，不偷偷改旧表。

本轮**4次真实训练、5次真实API、课程口径¥0.031819、133.500秒**。所有请求均有计量，原466条账本行完全不变；账本471/known¥8.047163/unknown16/pending0，无新增未知。数据包、模型、原始日志在忽略目录runs/sa8_live_01；课程正式test未使用。

下一步SA9单独预登记：按实际CSV含义比较等价recipe，保留真正的列名/ID/顺序/概率类别错误拒绝；复用V1/V3已取得的、绑定原包SHA和真实回复的契约，**不再发API**，只完成它们尚未开始的各4fit与交付。文本模型不重训。因此首轮＋恢复最多仍12fit；原始失败和恢复结果分开呈现，不将恢复假称首次无人干预成功。
