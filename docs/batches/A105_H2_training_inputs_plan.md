# A105：SMS结束后重建同一封存H2的训练侧输入

仅在A102 completed后执行，避免公共下载/图片解码影响当前C1计时。复用B的EuroSAT封存名单、BH-03构造器、两训练包及实际分区SHA；不重选样本，不训练/预测，不打开1000张地区留出或地理测试区图片。它是H输入准备，不选定最终策略，也不把H当开发性能集。

作者Zenodo记录https://zenodo.org/records/7711810已核对列出EuroSAT_RGB.zip和MD5 f46e308c4d50d4bf32fedad2d3d62f3b。按原B登记只下载94658721字节，SHA256 b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90。使用既有公共代理的进程级requests配置，不修改系统网络。单次请求、timeout60、无自动重试，下载总窗口最多300秒；失败保留partial和completion，不续传。

完成容器校验后才调用原prepare_b_h2_training_inputs.prepare，读取5000个已登记训练JPEG，生成两份2500训练图片包与六个内部工作文件。要求两包和内部manifest SHA与B原记录相同；新TaskSpec应真实启用9候选中的冻结视觉路线。所有模型拟合/网络在图片准备阶段禁止，0fit/学校API，预估准备1–5分钟；数据/权重/模型/源包不进Git，只归档协议和检查回执。

复跑：强Python执行python -m scripts.prepare_a105_H2_training_inputs --after D:/codex/python_withai/runs/a102_sms_C1 --assets D:/codex/python_withai/runs/a82_encoder_assets --output NEW_OUTPUT。原脚本/样本名单SHA绑定在本批protocol；H最终输入选择与策略冻结仍另行登记。
