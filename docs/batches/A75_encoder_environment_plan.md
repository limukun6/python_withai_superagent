# A75：冻结编码器运行资产，0训练/0模型API

A74合并核心真实交付已过14fit/60.365秒，累计A1228。现准备强方法依赖，不重跑旧效果矩阵，不读H/test。本机主环境Python3.14；另发现已有pytorch_learning为Python3.12、torch2.13.0+cu126/torchvision0.28.0+cu126。使用该解释器在仓库runs内新建system-site-packages venv，只继承只读torch运行库；所有覆写依赖装在新venv，原环境不改。

采用B已测锁定的numpy/scipy/sklearn/joblib/requests/Pillow/catboost和transformers4.53.3/tokenizers0.21.4/safetensors0.6.2；明确冻结huggingface-hub0.34.4。新环境不是B原CPython3.11的字节同环境，须单独验收，不能拿它直接宣称B结果已复现。pip只向venv装指定wheel，记录版本与继承来源，不把目录当可搬移的完整依赖包。

先验证安装，再从官方固定revision下载MiniLM三文件及官方PyTorch MobileNetV3-small文件；不发送凭证、不调用学校API，不使用其它BERT/SAM缓存替代。每文件按B代码已有完整SHA256和字节数验证，失败保留.partial、立即停，不自动重试/换镜像，不在模型运行时下载。预计约100MB模型+100MB以下科学计算wheel，torch大文件复用本机不下载；约5–20分钟，网络差则明确停止原因。全部在Git忽略的runs输出；只提交协议、摘要、可复跑代码。

若下载失败，仍可完成无编码器依赖的结构增量整合和契约检查；不能将缺资产候选标为已激活。成功后另预登记有限实训批次，验证真实特征/新进程交付，才决定正式菜单。

```powershell
Set-Location 'D:\codex\python_withai\runs\b_review_20261122'
& 'D:\codex\python_withai\.venv\Scripts\python.exe' -m scripts.prepare_a75_encoder_assets --base-python 'D:\conda\Anaconda\envs\pytorch_learning\python.exe' --output 'D:\codex\python_withai\runs\a75_encoder_assets'
```
