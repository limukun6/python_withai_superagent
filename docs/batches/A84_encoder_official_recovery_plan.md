# A84：复用真实编码器探针，只恢复正式入口

A81四真实worker全部成功：plain23.422秒、nonlinear25.094秒、preference16.078秒（交换概率误差1.11e−16）、vision10.719秒，各1fit；384/576维真实特征通过。A81在正式CLI启动前因验收runner清单将内部kind single_text误写为公开modality而拒绝，正式路径0fit；原completion保持stopped/4fit/92.969秒。不是编码器或网络失败。

只在新目录复制两个匿名训练/无标签包，CSV字节不变，清单modality修为schema允许的text；原包/runner/失败不覆盖。先实际load_training_task再训练。沿用A81同源码、权重、依赖、样本/seed/目录和5fit预算，执行正式local路径及新进程/源码包预测，不重复四个编码器动作。A起点1243，上限5fit，0API/H/test。不能把A81和恢复合称原一次无失败运行。

预计1–3分钟；source相等才允许恢复，首失败停，不自动重试。若赢家是编码器，验证缺权重拒绝输出与源码包依赖；若不是，明确不称完整编码器CLI交付通过，后续复用冻结worker模型另做无训练交付。
