# A126资产子批：只获取已交固定E5公开文件

新主线需要E5，A当前无该资产。使用新源ALTERNATIVES['e5']中固定的作者模型intfloat/e5-small-v2、revision ffb93f3bd4047442299a41ebb6fa998a38507c52，以及三个已交文件大小/SHA。公共下载走进程级127.0.0.1:7897，不修改代理配置；requests timeout60，不自动重试失败。

先保存runner/protocol，再逐个文件获取。以4MiB不重叠分段获取大文件，检查Content-Range/字节数；支持服务器首段返回完整200。三文件全部原始SHA匹配才ready。1800秒总窗口，失败保留partial与completion后停止。若作者revision不存在，原失败保留，须另外登记官方元数据核验和同字节恢复，不在同一批悄悄换版本。

本批0监督fit、0学校模型API；公共模型权重下载次数另计。权重不进Git，metadata/失败可审查。完成后离线匿名主线最多10实际监督fit，另有预先登记runner，不将下载成功当性能证据。
