# A113 原版本404：作者元数据核对

原A113单次请求config.json在B声明revision52f9d2dd711ffc3434e6cff17284b13a11629a90返回HTTP404，completion为stopped，0fit/API，保留原目录。不是校园模型API故障。不重发失败文件请求。

本小步仅GET作者Hugging Face模型元数据/api/models/sentence-transformers/all-MiniLM-L12-v2，timeout60，无自动重试，保存当前官方sha及原始元数据。若提供可用revision，再以单独恢复协议明确新下载revision；原B三文件大小/SHA作为硬门槛，不改变编码器身份。未经字节校验，不把新的revision自动写入生产代码或替代B性能证据。

此步不生成模型响应、不训练、不打开H任务。网络错误保留并停止，继续H1离线准备。
