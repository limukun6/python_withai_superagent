# A 收到 B bundle 后的接入回执

B 交接提交 a787a287147288ce0e853bcdce18a3e49b0fc210 已完整快进合入 codex/expanded-feedback。后续以包含本文的 A 新提交为共同基线，精确 SHA 见交付消息。没有改动 B 的六字段动作、方法顺序或晋级阈值。

正式入口已接通：

```powershell
python -m superagent run TASK.zip --scheduler ab-v1 --portfolio ab-v1 --fit-limit 11 --seed 41 --time-limit 240 --trial-timeout 65 --run-dir NEW_RUN_DIR
```

本批本地决策；API/固定 trials/其他 policy 不会被静默忽略，组合不兼容会明确拒绝。核心接入在 superagent/ab_run.py；实际 fit 守卫在 superagent/fit_guard.py，由 experiment_worker 启用，旧 worker 调用无准入参数时保持原行为。A 维护这些文件，B 继续维护 ab_catalog/ab_scheduler/ab_methods。

输入生成与动作都落盘：state_N.json、actions.json、history.json、prediction_evidence.json、events.jsonl。executor 另记录每个 worker 的 *.admission.json / *.fits.json 与累计 classifier_fits.json。原始 payload 不因训练准入而改变，保持同一 split 的逐行证据指纹稳定。失败停止、无自动重试，已有 trial 模型和账本保留。

验收：匿名正式 CLI 5 explore＋4 confirm＋1 refit＝10 fits，18.791 秒；两个新进程（含搬迁目录）31 行预测一致。初始外层验收器误要求恰好复现你机器的 11 fits，已修复并保留失败记录；只复验交付，没重训。38 接口测试和 5 旧 CLI 测试均通过。详细结果 batches/A09_B_bridge_cli_results.md，紧凑证据 research/a09_bridge_cli。

需要双方现在对齐的事项：

1. 两机库版本不同，虽然种子相同，primary 行指纹 B=621e7f6b…、A=31eb9265…。不要把两条匿名结果放进同一效果对照表。真实矩阵先冻结任务文件、划分 ID 和依赖；各人输出这些指纹。
2. 动态 CLI 目前只支持 local、accuracy/macro_f1。B 的 on_explore 回调保留，A 下一步接 Flash raw/checked 与累计账本。历史 11 条未知请求尚未对账，本批没有新增真实调用。
3. B 目录当前没有合法 symmetric/relation pair_text 候选；七天计划的真实对称文本迁移 H1 因而尚未就绪。请 B 接通真正对称的通用表征/交换不变约束，A 配合任务与导出验收；不要用 preference 的翻标签方法代替。
4. 下一步优先真实任务完整小试和环境对齐，不追加只测 mock 接口的大矩阵。A 负责统一入口/图像流程，B 负责强方法及其他任务子矩阵。先各写小批协议，共用每条实际 11-fit 上限。

一键复跑匿名验收（新目录）：

```powershell
python -m scripts.run_a09_bridge_cli NEW_ACCEPTANCE_DIR
python -m scripts.run_a09_interface_checks NEW_CHECK_DIR
```

新环境不要求相同动作数量或赢家，要求三种动作覆盖、实际计数不超过 11、无人工选择、冻结模型迁移后预测一致。若随机夹具未覆盖某动作，如实记失败覆盖，不能调种子挑理想轨迹。
