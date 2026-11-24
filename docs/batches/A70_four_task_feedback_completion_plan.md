# A70：补A2四任务同源固定历史，24真实Flash提案

用户已授权继续D3。先执行A69（已有B库本地四任务固定/动态），不与其并发训练；然后复用A69各任务seed707本地前三个探索的历史作为12状态，避免多训12次。coupon/preference/SMS/Bean各取前缀1/2/3，不看动作效果挑状态。原历史没有足够3个primary就停，不能挑别的成功路径。

每状态R/E共用同一训练事实、候选、预算、calibrated-v2输出契约；E只增加checks/memory。无profile，未发送原样本/标签/外层评分。第一anchor固定，提案从未试合法共同目录选。显式把本地state的llm上限改为3/已用0以构造一次独立提问场景，保留11-fit预算；本批不是完整在线循环。三个状态高度相关，不称独立任务重复。

24请求上限，每状态R/E各一次、顺序交替。requests timeout60、官方Flash名、显式campus_direct进程DNS保留Host/SNI/证书，必须获准联网的执行环境；失败即整批停，不重试、不换任务scope。共用唯一真实非空ledger，task1/task2/sms_spam沿用；只读scope清单已确认从无Bean历史请求，首次使用稳定dry_bean，登记原训练sha以免改名重置。旧11未知保留，incremental-research标课程总预算未核验。每任务两模式各.30元是预留/停止阈值非扣费保证，批次已知折算预计.15–.40元，总请求上限24。

全部提案冻结后，两新训练内seed30311/30321评价每个不同(task,candidate)；共享动作只训一次标去重，最多48新fit、0额外调用。与原A2预算一致，prepare复用已记账A69训练，不另收准备fit。行为与动作质量分开；没有自然错误机会就记不可测，不能说纠错率100%。自由文本语义未经独立标注不自动认证。

两份audit划分与A69任何三个内部划分实际行集合不同，预先保存SHA和行ID；只拿训练数据，H/test与外层标签不读取。A69源与A70同源，无新增生产修改。对照效果不限正向；不与旧A55拼成一次同源实验。标明方法契约与原最早A2设计有calibrated-v2差异；它是修订后的四任务开发机制实验。

运行：
```powershell
python -m scripts.run_a70_four_task_feedback prepare --study A69_OUTPUT --reference A47_OUTPUT --output NEW_PREPARED
python -m scripts.run_a70_four_task_feedback ask --prepared NEW_PREPARED --window-config WINDOW_JSON --ledger REAL_LEDGER --api-key-file KEY --execute --output NEW_ANSWERS
python -m scripts.run_a70_four_task_feedback evaluate --prepared NEW_PREPARED --answers NEW_ANSWERS --window-config WINDOW_JSON --output NEW_EVALUATION
```

phase都必须新目录；ask失败不自动再开ask；模型/数据/ledger不提交。正常预计Flash3–12分钟、动作训练1–5分钟；无服务时24×60秒为超时尺度但首次失败即停，留45分钟窗口。耗尽窗口不启动下一个配对。原始与修复/失败全部保留。
