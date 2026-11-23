# A95：非身份图像主线准备，0fit/0模型API

旧photo实际为Olivetti身份分类，后续不扩展该任务；旧记录只作历史讨论，不当自然花卉结果。复用B已登记的TensorFlow Flowers通用物体图像数据与确定性作者不交叉选样，不打开H。官方发布包228813984字节、SHA256 4c54ace7911aaffe13a365c34f650e71dd5bf1be0a58b464e5a7183e3e595d9c，来源https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz。

通过现有本地代理下载到新runs/a95_object_photos目录，timeout60、无自动重试，完整字节/SHA验证后才解析。用B原cohort/image_packet复用800训练/200留出、5花类、64×64 RGB、不同Flickr账号/照片ID及CC BY署名。原始图片和标签不进Git，只登记指纹/协议。此处只准备输入，不训练、不评分、不调用学校API，给下一批自然图像Q/C1使用；不能把准备成功写成泛化改善。预计1–5分钟，失败保留partial与completion。
