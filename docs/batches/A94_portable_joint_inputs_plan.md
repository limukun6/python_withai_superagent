# A94：双方实际训练行与分区可核对，0fit/API

B旧标题优化输入6629与A8301、Bean8714与A8710不同，单写同seed不能合表。只读A47已登记训练包，以当前共同源码/依赖冻结Coupon、偏好、Bean三任务seed707/808/909的实际train/search行号；导出B已有registered_partitioner能直接验证的v1 manifest（training_content_sha256、grouping、splits）。Coupon须与活动A89所引用A88实际索引一致，否则停。

另外记录有序每行内容SHA（含目标、无原始内容）、训练zip成员SHA、完整TaskSpec和baseline/full能力指纹。B可从同原始数据按指纹重建有序训练子集，再核对内容摘要/标签域/实际行号/菜单；内容不同不能以seed相同豁免。本文不用任务名决定算法，只给实验输入清单标识。

只在runs写二进制分区payload，Git仅JSON指纹与索引；不读任何val/H/test、不训练、不调用LLM、禁止网络和分类器fit。用B原工具反向验证三份manifest/各三分区，并与A88 Coupon关键索引逐项对齐。当前A89/排队A91源码不改。预计数秒至1分钟，偏好和Bean移交B跑整组C1，避免重复同一任务。
