# A-14：对称文本任务的输入契约（2026-11-21）

在A13照片路径完成后修改A所属schema.py，B方法库不动。发现的真实缺口：CSV pair_text读取器无论任务语义都要求pair_label_one=left，把同义判定标签1误绑定为左边获胜。

新增明确合法声明：pair_semantics=symmetric、pair_label_one=equivalent，标签只能0/1，1表示同义、0表示不等价，交换保持标签。保持preference/历史未声明语义时显式left的已有兼容；symmetric+left/缺失等价标签声明在训练前拒绝。不通过翻转标签临时适配，不用拼接文本假装交换不变。

本批只完成A任务读取/语义来源与正式入口无安全方法时的明确拒绝；B仍需真正交换不变的表征与方法。不会声称对称任务已经训练通过。以本地合成CSV执行加载契约、错误声明拒绝、TaskSpec保留语义/类映射、空目录不训练的正式入口，以及既有语义/目标检查。预登记0分类器fit/0API，产物protocol/results/completion；runner scripts/run_a14_input_checks.py。H1真实数据仍待公共源下载，不能把这些合成接口测试当迁移证据。

首轮25项全部通过，但资源审阅发现既有test_task_semantics内含1次DummyClassifier.fit，原runner错误标0fit。保留首轮原记录，标记其fit统计错误，累计计入这1fit。runner加入实际Dummy计数后新目录重跑（预留1fit）；输入/语义代码不变，不能将原轮称零训练，也不把这两次dummy当性能实验。
