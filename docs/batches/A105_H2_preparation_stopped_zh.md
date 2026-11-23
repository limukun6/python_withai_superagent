# H2准备尚未完成：公开原包下载超过预定窗口

A105在SMS整批结束后启动，向作者[Zenodo记录](https://zenodo.org/records/7711810)中的EuroSAT_RGB.zip发起一次公开下载。单次请求有数据传输，但累计下载窗口超过300秒，脚本按预登记停止；总耗时310.625秒，原.partial保留30408704字节，没有自动重试、续传或换来源。

原包未完整下载，未通过整包大小/MD5/SHA验证，所以没有运行图片解码或训练输入构造，不能宣布A本机H2已准备好。原source_ids封存名单及6000文件身份/六分区核对仍由A104提供，不受本次下载失败改写。

新增fit、模型预测、学校API、地区留出图像访问均0。公共下载没有读取API Key或修改真实用量账本；学校Flash在本轮SMS的12次请求均已成功。partial、原protocol/completion保留在runs/a105_H2_training_inputs，Git只归档元数据。

后续可复用队友已有的同SHA RGB原包或训练包；已有训练构造器scripts/prepare_b_h2_training_inputs.py接受本机--source和新--output并校验原作者MD5、原整包SHA和训练名单。若另行安排公开获取，应使用独立新批次/目录和明确窗口，不能冒充本次已完成或在后台循环重试。H的输入最终选择、策略与版本仍未冻结，留出性能继续封存。
