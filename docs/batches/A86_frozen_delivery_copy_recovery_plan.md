# A86：冻结编码器交付复核，损坏副本改普通复制

A85 nonlinear真实源码包与正式预测一致、缺权重拒绝已执行，但创建损坏副本时os.link被Windows文件权限拒绝，原批stopped/34.156秒/0fit/API，未推进vision，不声称损坏反例全部验收。原protocol/runner/log保留research/a85_frozen_delivery_initial。

本批独立新目录，沿用A85两冻结模型和全部验收，不重训。唯一执行准备变化是将权重副本用shutil.copyfile复制，避免硬链接权限和共享内容风险；原权重不修改。新增磁盘约100MB，可接受。batch/plan显式A86，新runner哈希不覆盖旧身份。0fit/0学校API，A累计1247不变。预计1–3分钟。
