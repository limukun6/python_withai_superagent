# A54：不依赖原仓库目录的模型交付

在导出前登记，0训练/0API，不改冻结生产源码。A44/A52的搬迁仍能从原仓库import自定义模型类，因此不能等同完全独立的预测包。本批导出模型+TaskSpec+selection+确切superagent源码+依赖版本+predict.py，排除数据、API Key、SQLite、原日志和.git。

固定导出A47五个full目录策略模型（去重任务用shared）及A44最终coupon故障partial模型，不因分数换来源。模型/源码SHA与run_manifest冻结一致，否则拒绝；partial原标记保留，不变成complete。输出为本机zip，二进制不进Git。

使用当前venv的python -I，在与原仓库不同的工作目录执行解包后predict.py，显式只加载包内src，报告实际superagent模块路径确在包内；禁网络/分类器fit，与原预测CSV逐字节一致。五真实类型+一个partial，共6个预测包。只用已有features，不读标签/重评分。依赖用本机实际版本生成并校验；本批验证隔离的源码交付，**没有验证在空环境联网安装依赖**，不写成已做全新机器安装。

最多6次模型复制/压缩与6次新进程推理，0fit/API；串行推理尽量在live训练队列结束后执行。代码tools/export_model_bundle.py和scripts/run_a54_export_acceptance.py，输出a54_selfcontained_delivery。最终模型冻结后可复用相同导出器，研究模型包不是最终提交版。
