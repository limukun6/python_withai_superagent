# H2 · 图像迁移数据的预登记

H2 选 EuroSAT 的 RGB 土地覆盖图像，共十类，来源为 Copernicus Sentinel-2，由 Helber、Bischke、Dengel 和 Borth 整理。A 的图像开发任务是手写数字。旧实验已经使用过的 Fashion-MNIST 和 Kuzushiji-MNIST 继续归入开发材料。EuroSAT 作者将数据按 MIT 许可发布，并要求留意 Copernicus Sentinel 使用条款。本仓库交付来源、文件指纹和行号，图像留在本机。[作者说明与许可](https://github.com/phelber/EuroSAT) · [作者数据版本 v2](https://zenodo.org/records/7711810)

原始 64×64 RGB 文件按 Zenodo 公布的大小和 MD5 校验，再计算 SHA-256。地理划分使用 TorchGeo 固定版本的训练和验证两个清单，按发布的 SHA-256 校验。该划分按经度分区。未来训练只用训练地带，外层评价只用验证地带。正式地理测试清单和对应图像均不读取。[地理划分实现](https://github.com/torchgeo/torchgeo/blob/main/torchgeo/datasets/eurosat.py) · [固定来源](https://huggingface.co/datasets/torchgeo/eurosat/tree/1ce6f1bfb56db63fd91b6ecc466ea67f2509774c)

为适应小组的单机预算，预先选两组互不重叠的图片。每组每类训练 250 张、外层 50 张，即每组 2,500 / 500 张。每个地理区域和类别内按固定盐与原文件名的 SHA-256 排序，依次取样。先登记训练区域再登记验证区域，同像素图像只保留第一次出现的条目。训练及留出的文件名在每个分区内按同一固定规则排序。每组训练输入再按生产代码登记内部 707、808、909 三个去重划分。

这一批只解码图像确认格式、检查跨分区重复、登记确切文件名并汇总类别数。允许训练、预测、评分、模型 API 调用均为零。两个划分属于同一数据源，最终解释限于本次抽样和地区。冻结唯一部署策略后，按共同 H 协议打开训练和评价。
