# BV-01 · 在真实 RGB 照片上验证通用视觉候选

使用 A 的公开提交 `0418ae1` 合入 B 既有方法，保留 A 的正式入口和固定/动态分派。B 负责视觉候选的资格验证。本批是自然照片开发实验，A 的最终图像 Q/C1、以及封存的 H2 仍按二人共同计划执行。

**样本在训练前锁定。** TensorFlow Flowers 官方训练包共 3,670 张五类照片。原包 228,813,984 字节，SHA-256 为 `4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c`，按包内的 CC BY 2.0 授权及逐图署名使用。每类按固定盐与相对文件名的 SHA-256 排序，类别按字母顺序轮转取样，全局每个已标识的 Flickr 账号及照片编号最多一次。每类取 200 张，按取样顺序前 160 张供优化、余 40 张留出，再用另一固定盐混排行序。两边均匿名化类别和图片路径，衍生图片包附对应作者与来源署名。实际解码后全体 1,000 张不应有重复像素。输入只给优化进程 800 张有标签照片，冻结后才在 200 张照片上评估。

两组均用正式 `superagent run --scheduler ab-v1 --portfolio ab-v1`、内部种子 707、同一源码和数据、相同可用依赖、最多 11 次实际分类器拟合（含重训），每组 1,200 秒、单次 300 秒，单 CPU 机器依次运行。先跑原有像素和梯度方法，再跑同一目录增加已校验的官方 MobileNetV3-Small ImageNet-1K V1 权重。组内从第一步起自主决策，两个模型全部冻结后统一评分。若未选中新增方法，也保留实测轨迹。真实 API 请求上限为零，资源下载在开发准备阶段完成。

统一图片输入为 A 的 JPEG/PNG 文件接口，64×64 RGB 双线性解码。MobileNet 使用本地完整权重摘要校验、官方 256/224 预处理、全局平均池化的 576 维归一化向量、单个逻辑回归分类器。无梯度更新、GPU 训练或外部推理请求。对照组保留原有梯度候选。两组均独立核对真实 fit、候选全集、三份内部行序、标签和样本指纹。将两个冻结模型分别在新进程和搬迁目录导出同一批预测；比较逐行变化、准确率、耗时和实际拟合数。一次外层划分只说明本开发样本上的表现。

[官方数据说明](https://www.tensorflow.org/datasets/catalog/tf_flowers) · [发布包摘要](https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/datasets/tf_flowers/checksums.tsv) · [PyTorch 权重与预处理](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)
