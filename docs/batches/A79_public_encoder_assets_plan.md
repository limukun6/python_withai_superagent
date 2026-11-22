# A79：公开依赖显式进程代理，学校API不变

A75原环境复制下载43–52KB/s，实际Python公开HTTP代理映射为空；下载到100MB CatBoost前已核对并只停止本批pip PID9892，原completion保留stopped，0fit/0模型API，原runner保留research/a75_encoder_assets。现单独新目录runs/a79_encoder_assets，明确使用已监听的http://127.0.0.1:7897，仅供公开PyPI/HuggingFace/PyTorch下载；不改系统代理、NO_PROXY、DNS、学校API调用或防火墙。

沿用A75精确依赖与四权重完整SHA256，复用只读已装torch2.13/vision0.28。pip --retries0/--timeout60，HTTP timeout60，失败本批停止，不自动换镜像或重试。每文件.partial验证后才rename。各批独立新目录，不把取消的A75包装为成功；不清旧账本。预计代理可用时3–10分钟，不以预测冒充完成。

只准备依赖/权重，0fit/0LLM请求/0学校费用；A累计1239不变。完成后须另写真实编码器批计划，才允许训练/推理。原B使用CPython3.11，本机覆盖环境3.12应单独记录并验收。权重、venv不进Git。

```powershell
Set-Location 'D:\codex\python_withai\runs\integration'
& 'D:\codex\python_withai\.venv\Scripts\python.exe' -m scripts.prepare_a75_encoder_assets --batch A79 --plan 'docs/batches/A79_public_encoder_assets_plan.md' --public-proxy 'http://127.0.0.1:7897' --base-python 'D:\conda\Anaconda\envs\pytorch_learning\python.exe' --output 'D:\codex\python_withai\runs\a79_encoder_assets'
```
