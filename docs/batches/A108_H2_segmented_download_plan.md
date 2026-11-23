# A108：公开原包分段获取与 H2 训练输入验收

A107 已结束，247.781 秒后长连接被提前关闭，错误为 IncompleteRead，保留 48,234,496 字节，未打开图片。它不是 1800 秒上限触发。原批不覆盖、不重新运行。本批针对长流中断，改为有界 Range 分段请求；不改变系统网络配置。

从 A107 保存的 prefix 开始，每段最多 4 MiB，按预定偏移顺序单次获取，共需 12 段（以实际余量计算）。每段严格要求 HTTP 206、准确的 Content-Range、相应字节数；失败立即停，不自动重试该段。仅完整合格段才写入新的 partial 并登记 SHA。timeout=60、总下载窗口1800秒，块内每次读取检查窗口。接收中失败的块单独留为 segment.partial，不污染已接收前缀。原 A105/A107 文件保持不变。

之后的原包 SHA/作者 MD5、训练成员访问白名单、5,000 图像解码、两训练包及六分区指纹、full9 目录验收沿用 A107 方案。0 拟合、0 预测、0 学校 API；下载是公开文件 HTTP，不写模型 API 账本。预计 8–20 分钟下载，加 1–5 分钟输入构造。失败保留 protocol/completion/分段进度。

代码独立保留为 scripts/prepare_a108_H2_segmented.py，原 A107 脚本不改动。在 integration 工作树执行：

```powershell
& 'D:\codex\python_withai\runs\a79_encoder_assets\venv\Scripts\python.exe' -m scripts.prepare_a108_H2_segmented --prefix 'D:\codex\python_withai\runs\a107_H2_local\EuroSAT_RGB.zip.partial' --assets 'D:\codex\python_withai\runs\a82_encoder_assets' --output 'D:\codex\python_withai\runs\a108_H2_segmented'
```

本批准备输入，不选择最终 H 策略、不评估留出，不接管 B 的 H2 实验分工。所有原包、图片、worker 文件均在忽略目录，Git 只发布脚本与脱敏证据。
