# A72：不要把未绑定的历史回顾当当前模型声明

A70后验机器审查有两条classifier矛盾flag：coupon_3 E的“Logistic C=0.05 is materially weaker.”、bean_2 E的“extra trees leaf1 showed high train accuracy but lower search accuracy.”，都在review中回顾已有旧模型，不能默认是当前选中的CatBoost/Logistic。它们是审查器的引用歧义，不是已独立标注的模型错误。

本批先保存旧审查flag，修改通用范围规则：review/assessment.interpretation内没有显式当前/选中对象、没有trial/candidate引用的classifier词保持unverifiable，不能既强确认也强否定；明确选中对象仍按执行事实核验，结构化configuration_claim不放松。不是新增两个特定词句的豁免。

0fit/0API，A累计1214不变。新6项无训练反例+原prose/explicit配置检查，旧48开发集不改预期复跑；重新审查A70输出只作后验侧车，不改原回答或原反馈内容、不得据此归因A70动作质量变化。生产源会更新，旧A69/A70/A71模型/结果保留原源指纹，不冒充新版正式结果。
