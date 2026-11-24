# A107：A 本机恢复 H2 训练输入准备

用户于 2026-11-23 明确要求 A 自行解决下载/输入准备。复用 A104/BH-03 已登记身份，不更换数据源、样本或分区。此批只准备输入；不选择最终策略，不运行 H 性能实验。

- 保留 A105 失败目录不变。新目录单次请求官方 URL，尝试从旧 partial 的字节偏移续传；HTTP 206 必须严格匹配 Content-Range，HTTP 200 则只用本次完整响应，不拼接。任何其它状态、异常或校验失败停止，无自动重试。
- timeout=60 秒，下载总窗口 1800 秒（阻塞读取最多额外 60 秒）；写入流检查上限 94,658,721 字节。过程每 30 秒记录进度，便于接手。
- 整包 SHA-256 必须为 b4f5b234ecb7d7ff9c6cddb046543b4717c53fd6e9815be6c0e80cc614f51b90，并通过原作者 MD5，才能打开 ZIP 图片成员。
- 仅允许读取名单中 5,000 个训练 JPEG，额外包裹 ZipFile.open 记录并拒绝名单外成员；不解压或解码留出图片。底层原包字节哈希不等于打开留出图片。
- 使用 B 原构造器生成两份 2,500 行训练包与六个内部划分，要求 train_archive_sha256、inner_manifest_sha256 与 BH-03 完全相同。核验正式入口能识别图像契约和冻结视觉候选。
- 新增分类器拟合 0、预测 0、学校 API 0；预计下载 10–20 分钟，输入准备 1–5 分钟。最终记录实际耗时；失败也保留 protocol/completion，不覆写旧批。
- 数据、原包、worker 文件留在忽略目录；只归档脱敏元数据。当前 B H2 执行分工不变，本地准备成功不等于 H2 已验证。

复跑（在 integration 工作树，output 必须为新目录）：

```powershell
& 'D:\codex\python_withai\runs\a79_encoder_assets\venv\Scripts\python.exe' -m scripts.prepare_a107_H2_local --prefix 'D:\codex\python_withai\runs\a105_H2_training_inputs\EuroSAT_RGB.zip.partial' --assets 'D:\codex\python_withai\runs\a82_encoder_assets' --output 'D:\codex\python_withai\runs\a107_H2_local'
```
