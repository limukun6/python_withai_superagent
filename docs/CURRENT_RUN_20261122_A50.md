# 当前状态：本轮队列已完成

A50/A51/A55/A58与所有收尾检查已完成，无训练/API队列运行。下文是历史运行记录。最新结果见`batches/A37_A59_round_results_zh.md`，后续计划见`NEXT_RESEARCH_AFTER_A58_zh.md`。A累计970fit，真实ledger275行/已知估计¥5.9765785/旧11 unknown/pending0。不要重复启动；H/test仍封存。

# 当前正在执行A50/A51：接手先看实际状态

用户要求连续推进3小时、不停下。原23:24–02:24窗口被Windows 00:21待机/08:32唤醒打断；原A32保留8完整、1partial、11未启动，91fit/20真实成功聊天/¥.163567，无外层评分。A34/A35/A36旧队列未启动。不能将它们写成完成。

## 恢复与当前实跑

恢复窗口另记主仓库runs/recovery_window_20261122/window.json，08:42:29–10:42:29+08。A42只接原未启动项，1基线fit后DNS失败、没有重试；绑定原ticket审计not_dispatched，原unknown转移保留。执行层对照进一步查明沙箱拒绝网络，授权环境Python又拿到198.18.0.69虚拟IP。不能把这两层都说成学校API挂了。

scripts.campus_direct是显式进程解析，固定官方域名→已知10.121.10.250，不改系统/代理，保留Host/SNI/证书。09:11左右a50_models_direct的带认证models实际200、19模型、Flash存在。不要再重复诊断/聊天连通小测。

当前工具会话36542（不是Windows PID）运行临时防空闲待机包装器，内部scripts.run_a50_live_queue依次：

1. a50_facts_calibration：两课程开发任务×profile off/train×strict/calibrated，8路径，11fit/3请求，所有模型冻结后才读指定val。
2. A50成功且余≥50分钟时，a51_joint_transfer：SMS 00→11、photo 11→00，4路径。联合结构跨输入，不分拆两因素。

真实ledger为主仓库runs/a23_real_local_history/ledger.sqlite。A50开始224条/known5.5403165/原11unknown/pending0。任何新增unknown/pending按既有规则停；失败不重试，不重开旧路径，不换空账本或scope。课程总预算仍未核验，实际平台扣费null。当前输出a50_live_queue/stages.json、a50_live_queue/A50.log、a50_facts_calibration/checkpoint.json、各run usage与completion；不要仅凭进程存在声称在训练。看到A50状态completed才能跑summarize_a50_factors。

## 冻结区

本队列运行期间不改superagent全部生产文件；scripts/run_a50_facts_calibration.py、campus_direct.py、run_a29_choice_matrix.py、run_a34_transfer_coverage.py、a29_worker_observer/sitecustomize.py。可写只读分析、独立tools原型、文档。不要并发训练，回归中7fit也等正式训练队列结束再跑。

## 已完成新工作

- A40原中断coupon模型partial导出，6494/8117训练行、2030预测，新进程一致，0fit/API。
- A44–A46：正式入口增加三个默认off/旧契约的显式开关：task-profile train；proposal-contract calibrated-v2（Flash专用）；partial-delivery checkpoint。五真实任务5完整＋5人工故障全部验收，25fit/142.736秒，十模型搬迁预测相同；外层标签未开、0API。原故障返回2且requested=false，partial不是完整成功。
- 两次先前失败验收3+5fit保留，修了摘要变量覆盖和误读checkpoint.kind；恢复损坏证据0fit探针1合法+6拒绝通过。另此前124回归7fit。不能只报成功25而漏8失败。
- A47同源方法库Q：五任务7物理路径去重、67fit/328.799秒，Coupon .738916→.750246（+1.133pp），photo .975→.975；preference22/22、SMS6/6、Bean5/5目录相同，只有shared路径，不是独立0收益实验。七模型重载通过，H/test未开。research/a47_catalog_Q。
- A48模拟整条v2/profile/SSE链：3用例9fit/5模拟HTTP/26.946秒、0真实API，真实ledger SHA不变；两成功raw/E共同上下文逐项同，timeout一次后partial，0重试。模拟ledger不能导入真账本。
- A49实际参数观察独立原型：五已有模型读取参数、五人工错误声明均能识别、真实原声明无冲突，0fit/API；尚未接生产，不能说已提高选择效果。
- A44重建24个旧默认请求相同，另24个profile+v2 R/E公共字段相同，0fit/API。新源修复后最终运行的默认off还由A47所有真实state检查无profile字段。

A50之前累计本A跟踪窗口812fit＝原597＋A31 7＋A32 91＋A42 1＋A44初回归7＋失败3+5＋成功25＋A47 67＋A48 9。这是被跟踪的A窗口，不是全项目/队友总数；历史40fit回溯更正已含在原597内。之后加A50/A51实际fit及最终回归真实7等，不把上限当用量。

## 收尾还要做

等待/检查两个live阶段，失败则继续不依赖它的离线整理，不发新请求。用summarize_a50_factors分别归档真实A50/A51；A47已归档。最终回归需串行，源不能中途修改。做模型搬迁、图/表、来源/fit/token/费用汇总和下一轮计划；更新EXPERIMENT_PROGRESS与A_LINE_RESUME，实际key字节扫描、git diff --check通过后提交推送当前codex/expanded-feedback。

不要复跑H1/H2或正式test。没有实际completion的A38四模型搬迁、旧A34/A35/A36不能补成已完成；若用新A50/A51做新四模型验收，先登记来源与规则，不能冒充旧A38。
