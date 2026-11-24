# A109：固定 LF 重建 H2 训练输入

A108 原包已通过 SHA-256/作者 MD5；12 个分段完成。两训练 ZIP 与 B 完全相同，六个 worker/prediction-label/row 指纹均相同，但 inner_splits.json 原始字节 SHA 不同，因此 A108 按协议停止。只读诊断显示：两文件仅将 CRLF 替换成 LF 后，其 SHA 就等于 B 原登记值。原失败、原文件全部保留。

本批完全离线，复用已校验原包。新输出目录重建，唯一变化是在调用 B 原 prepare 构造器时，将 fixture.write 临时包装成相同 JSON 序列化、UTF-8、固定 LF 的字节写入。不改生产模块或原 B 脚本，不事后修改旧结果中的 SHA。构造完成后严格比较两 ZIP、两 manifest 及全部六 worker 登记字段，正式 registered_partitioner 再读取固定 LF 文件核验。

只允许打开原登记 5,000 张训练图。白名单保护且记录访问数量，禁止打开其它原包成员；预期 1–3 分钟。0 新训练、0 预测、0 网络/API。验证9候选图像目录及本地冻结权重资格。原包、数据、worker 不进 Git，仅归档回执。H 策略/性能仍未冻结和运行。

复跑：强 Python 在 integration 下执行 `-m scripts.prepare_a109_H2_canonical --source D:/codex/python_withai/runs/a108_H2_segmented/EuroSAT_RGB.zip --assets D:/codex/python_withai/runs/a82_encoder_assets --output NEW_DIRECTORY`。
