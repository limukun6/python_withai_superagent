# A142–A144：新真实数值任务的冻结流程迁移

用户在上一轮三项补证完成后回复ok，A继续补真实任务迁移。起点00b16f2，A累计1614fit，真实ledger372/known6.8218955/unknown12/pending0，A独占空闲。原12未知继续授权有效，不等官网账单。先保存本计划与可复跑代码，不修改94个生产源，不按新任务效果加方法/调序/换阈值。

## A142 输入登记，0训练/0模型API

选UCI Banknote Authentication，正式页https://archive.ics.uci.edu/dataset/267/banknote ，DOI10.24432/C55P57，CC BY4.0，1372行/4连续特征/二分类；项目docs/research/scripts检索未发现旧开发记录，不保证LLM预训练未见公开数据。它是预计算特征的数值任务，不是原图像迁移。只依据官方规模/输入/许可选择，不查分类基准成绩。

原包https://archive.ics.uci.edu/static/public/267/banknote%2Bauthentication.zip ，公共下载走已有本机7897代理，不改代理配置；最多一次请求、timeout60，下载失败保留，不自动换来源重试。只解析指定数据TXT，核1372×5、finite、0/1标签，保存整包/成员SHA。行级完全相同特征作为同组，外层按现有grouped_partition固定71424取第一合法折，不试多个划分挑分；组不跨边界。训练包匿名，只给train.npz和manifest；查询只有外层X、无标签，y存另一私下joblib。

先登记训练内707/808/909实际行号并按当前任务内容签名，预备完整合法full目录，numeric-kernel/置信路由/参数语义均off，不因Bean或H1成绩改默认。不能凭公开数据库有全部标签就让选型器读取外层。

## A143 两完整自主路径，最多22fit/3Flash

同94源、同原训练行/内部划分/候选/11fit含refit、单fit300秒/路径1200秒。L0＝local＋ab-fixed＋full/standard；EC0＝checked-history＋compact＋calibrated-v2＋ab-fixed，同候选。显式研究配置EC0，不事后替代原F*=L0；差异不单独归因压缩、检查或LLM，因为本对照改变整套决策方式。A140才是同历史full/compact机制小试。

顺序L0→EC0，两模型全部封存前不评分外层，不根据L0分数改EC0。各4探索/6复核/1refit，Flash至多前三次新候选请求，实际记账。EC0沿同一实际ledger和accepted-history原12ID清单，task scope固定banknote_authentication，单路径估计停止阈值¥.30，无新空账本、无自动重试，新增unknown/pending或失败立即停该付费路径。保存失败/partial，不能补跑追成功。新fit最多22，A最多1636；新请求最多3，全部成功时ledger最多375。

## A144 全封存后交付/评分，0fit/0API

先统一核两模型完成、来源/菜单/行号/真实fit及SHA，再解封外层标签。native/搬迁/隔离包预测逐字节核验，主指标accuracy，报告答对数、macro-F1辅助、训练多数类参照、改对/改错及行序，不用外层调参。报告实际训练时间、规划等待、tokens/费用和最坏退化；单次顺序时间不作稳定加速因果结论。

这是一个新项目任务、一个划分，不能当多个独立任务、最终课程Unseen或任意分布保证。完全相同行分组无法替代缺失的纸币实体ID；保留实体独立性未知。若两路相同或退化，照实保留，不挑其它数据集或种子追正增益。不影响B继续H2/Released2、Task1旧候选或原C1版本。

预计小数据CPU训练/交付数分钟，含开发与API约15–30分钟；两路径硬上限40分钟加进程裕量，不把预计当保证。数据/模型/weights/ledger/key不进Git，只提交代码、协议与脱敏结果。遇额度中断可从独立阶段继续，完成路径不得重复训练/调用。
