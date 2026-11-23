# A 本机 H2 输入准备已完成

2026-11-23，A109 completion 为 prepared。本机可以独立准备与 B 相同的 H2 训练输入，无需等待 B 传图片包。当前仍待共同版本、策略和预算冻结后运行 H2 性能实验。

## 瓶颈、处理及证据

| 瓶颈 | 处理 | 实验与结果 | 证据边界 |
|---|---|---|---|
| 公开文件慢下载和长连接中断 | 严格 Range 校验的 4 MiB 分段获取；每段失败就停 | A107 长流在247.781秒提前关闭，保留48,234,496字节；A108新批12段、324秒完成余量，整包94,658,721字节、SHA和作者MD5正确 | 文件传输修复；没有修改学校 API 或系统代理 |
| Windows 文本换行影响跨机指纹 | UTF-8、固定 LF 字节写入 | A108两ZIP/六worker相同但manifest原始SHA不同，按协议停止；只读诊断确认LF后正好等于B登记SHA；A109从原包重新构造，全部原始指纹通过 | 没有忽略校验或事后把失败标成通过；A108原件保留 |
| 准备训练输入时可能误读留出 | 原包图片成员白名单、访问审计、离线运行 | 仅打开登记的5,000张训练图；六分区重读通过，留出/空间测试图片打开0 | 原包字节哈希会读取完整压缩文件字节；没有解压/解码留出成员或计算其性能 |

A109耗时83.344秒。两组45101/45102各2,500训练行，三份内部划分各2,000拟合/500内部评价，共六份。两训练ZIP、两inner_splits.json、六worker的全部登记字段与BH-03一致，正式输入契约启用9候选，包括本机冻结MobileNet特征路线。未加载网络做特征提取或预测。

本轮A107/A108/A109新增分类器拟合均0、学校模型API均0、模型预测均0。A累计1501次拟合不变；没有写API账本。分段公开文件HTTP请求不属于模型API用量。A107/A108均保留stopped记录，不覆盖失败。除A105历史耗时外，本次三批控制器耗时合计约707.578秒；其中A108下载324秒，后验LF诊断/文档时间另计。

## 本机入口和交接

- 原包：`D:/codex/python_withai/runs/a108_H2_segmented/EuroSAT_RGB.zip`。
- 已验收训练包：`D:/codex/python_withai/runs/a109_H2_canonical/training/45101/train.zip` 与 `45102/train.zip`。
- 对应目录的 `inner_splits.json` 和 `worker_inputs` 为同登记分区；不能用另一随机划分冒充。
- 完整回执：`D:/codex/python_withai/runs/a109_H2_canonical/completion.json`；Git脱敏证据为 `research/a109_H2_canonical_inputs`。
- 失败证据：`research/a107_H2_download_stopped`、`research/a108_H2_segmented_download`。A105/A107 prefix保持不变；所有源包、图片、worker都留在忽略目录。

可复跑代码与先行计划分别为 `scripts/prepare_a108_H2_segmented.py` / A108计划、`scripts/prepare_a109_H2_canonical.py` / A109计划。后续不需要重新下载。跨平台构造登记JSON应显式固定LF；目前通过构造器临时包装完成，没有改B原脚本或共同生产控制器。

H2准备完成只证明输入可复现及方法入口可用。它尚无模型成绩，未选F*、未冻结最终H策略，B的H2负责范围保持既有对齐。下一步收齐B新提交及当前C1结果，确定共同版本/策略/预算后再运行迁移，不把训练输入验收计为通用性能证据。
