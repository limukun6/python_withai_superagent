# A90：冻结编码器可运行，不等于课程例外已确认

**A93已解决本页当时的不确定性**：用户转来老师明确答复，限制针对SuperAgent语言模型，允许通用下游预训练特征、由Agent自主选用。原文见TEACHER_ENCODER_CLARIFICATION_20261122_zh.md。默认现为auto，本页restricted/待答复叙述保留为历史，不作当前阻断条件。

原始要求来自用户提供的AIAA3102 Final Project Full Assignment Requirements第4节：指定日期前Qwen/DeepSeek模型、官方标准token计价与本地部署同样计费。该段未明确说明冻结MiniLM/ImageNet特征是否豁免；目前未找到老师明确例外答复。已向用户问原文，未把未回复或预选项当同意。此前把技术可执行性直接写成正式运行建议，缺少资格边界，现修正。

课程默认环境入口`run_with_encoder_assets`现在默认`--encoder-policy restricted`，清空两类权重变量，沿用同一个superagent CLI/方法注册/调度；无需先下载编码器。可借已有Python3.12环境，但不会启用预训练特征。`--encoder-policy research --assets ...`才显式启用原研究路线，不由此认证课程许可。核心superagent未改，正在跑的Coupon C1不使用这些编码器，源码/目录/预算不变。

0fit/0API检查通过：restricted无资产、restricted借环境、research有资产三种dry-run；research无资产拒绝；真实help转发。只读目录快照：

| 任务 | baseline | 课程默认full | 资格待定研究full |
|---|---:|---:|---:|
| Coupon | 14 | 15 | 15 |
| 偏好 | 22 | 30 | 32 |
| SMS | 6 | 14 | 16 |
| 照片 | 7 | 8 | 9 |
| Bean | 5 | 5 | 5 |

词法/字符/SVD/有界组合、类别交互、梯度等常规方法不依赖这些待核实编码器。A81/A84/A86证明的是技术执行/交付，暂作为开发研究；A88照片full实际试过冻结视觉，故在资格明确前也保留开发身份，不能只因最终选中梯度就称搜索全过程无预训练模型。

本地编码器的前向不等于学校API调用；“新增学校API费用0”也不代表它们已获许可或能按课程规则计为全部模型成本0。其资格、所需token计账范围和计价依据未确认，不认证这部分课程预算合规。原Flash真账本与历史11未知原样保留，Coupon新增调用另计。

接下来在C1队列结束后补仅梯度扩目录photo full8，复用同源合法baseline7，另预登记有限预算，不重跑已有Coupon。若后来有课程明确说明，再登记适用范围；不回填当前资格已确认，也不重写旧失败/结果。
