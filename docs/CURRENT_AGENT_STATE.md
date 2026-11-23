# 当前接续状态：2026-11-23

本卡是当前状态；EXPERIMENT_PROGRESS与各CURRENT_C1/Q旧快照只作历史。继续前核对真实completion/physical_fits/ledger，禁止重复启动已结束批次。

## 已完成

- A89 Coupon六组C1完成：53新fit＋11预登记复用、12成功Flash、56.8分钟。L0/L1/E0准确率75.0246%，R0/R1/E1为75.2217%，净差4/2030。核对反馈没有稳定胜过原始历史。65966输入/14405输出tokens，新增已知估计¥0.123586。详见batches/A89_C1_results_20261123_zh.md及research/a89_coupon_C1。L1旧时耗不能当新效率因果证据。
- A91旧photo消融完成10fit，两个旧防待机护栏均已释放。旧photo实际Olivetti人脸身份分类，后续不再扩展，不当作自然花卉证据。
- A94三任务九分区内容/行号摘要可交接B，0fit/API。Task2 8301行/32候选，Bean8710/5。见A94_TO_B_PORTABLE_C1_INPUTS_20261123_zh.md。两数据版本不同不能直接拼表。
- A95花卉对象数据准备191.281秒、0fit/API；原TF Flowers包SHA/大小、1000作者与图像去重通过，800训练/200开发留出，B已见开发队列，不是H。
- A96三路径完整自主Q完成32fit/300.438秒/0API，A累计1353→1385。baseline7与无预训练8均42.5%，full9自主选MobileNet特征达87%，净多89/200。三个模型先冻结再评分，新进程预测一致。生产source未变，临时睡眠请求已随控制器退出释放。见batches/A96_flower_Q_results_20261123_zh.md、research/a96_flower_Q。

## 接下来执行A97

A97花卉完整9候选六组C1已实际启动，入口session66039（仅本会话有效），代码/计划已推送c64bee4。输出D:/codex/python_withai/runs/a97_flower_C1；先查completion/checkpoint，禁止重复启动。显式复用A96 full的10旧fit，另外五路径最多55新fit、12Flash，增量估计阈值合计¥1.20。相同source/菜单/内部分区/11fit/3调用/300/1200；L0/R0/E1/E0/R1顺序，全部模型冻结再统一评分，已见开发数据不称新泛化。真实账本当前唯一写入方A，B先跑本地组，付费写入权以明确交接为准。

运行协议batches/A97_flower_six_arm_C1_plan.md；runner scripts/run_a97_flower_C1.py。预检独立输出a97_flower_C1_preflight（0fit/API）；若更新核验逻辑，独立v2目录，不覆盖原记录。正式使用强Python与已校验assets；临时防待机上下文结束释放，不改持久电源设置。活动时不修改superagent或protocol.script_hashes内helpers。

A98已在隔离工作树runs/a98_prose_scope修复自然日志的历史baseline误报，分支codex/prose-baseline-scope-20261123已推送c62c47a。16开发例6→16通过、A89原12提案重放机器矛盾1→0（消失项是开发者复核误报），47相关回归通过，0fit/API。A97未结束前禁止合入，不能改本轮source；之后可整合并记录新版本。B本轮C1必须固定c64bee4生产内容，不把后续修复HEAD与旧组拼同源表。16例不是独立48/24语义验收。

## 资源与边界

- A97前A累计1385。真ledger311请求/known¥6.2653195/历史unknown11/pending0；每次先读真实ledger，不用此快照覆盖新数据。唯一ledger D:/codex/python_withai/runs/a23_real_local_history/ledger.sqlite，密钥根api_key.txt，均忽略、不提交。
- 用户已授权incremental-research，历史11条保留未知，不再等平台记录才研发；strict默认仍拒绝未知，课程总预算/实际扣费不能认证。新增unknown/pending或请求/训练失败立即停，不重试、不清账本。A97检查意外并发账本写入。
- 老师明确允许通用下游预训练特征，原文TEACHER_ENCODER_CLARIFICATION_20261122_zh.md。默认wrapper auto，有校验资产提供编码器，无资产常规fallback；restricted仅消融。LLM仍DeepSeek/Qwen、任务名不路由。
- 生产superagent内容自aa47bb9起未变。强Python D:/codex/python_withai/runs/a79_encoder_assets/venv/Scripts/python.exe，assets在runs/a82_encoder_assets；普通环境仍根.venv。学校API timeout60，既有campus_direct包装器固定官方内网解析且保留TLS，不改系统代理。
- H1/H2与课程test封存；独立48例和24语义标注仍缺，不能宣布D3全面通过。B负责Task2/Bean完整C1；A的花卉是已见开发数据复现，两边都要报告作用范围与新进程交付。
