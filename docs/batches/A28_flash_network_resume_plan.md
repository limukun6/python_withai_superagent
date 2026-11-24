# A28：网络调整后的Flash连通与真实决策小试（2026-11-21）

用户明确“现在应该可以了”，授权网络调整后新批测试。起点202a250，不修改代理/hosts/DNS/防火墙，不自动重复上一失败批。H1/H2继续不训练不评分。

## 执行顺序与上限

1. 用已有scripts/test_campus_flash_once.py在新目录a28_flash_connectivity执行1次0训练SSE连通请求，正确模型DeepSeek-V4.1-Flash，同一真实累计ledger.sqlite、sms_spam范围、incremental-research。预检在该脚本内先于读取key/请求；任何新增unknown/pending即停止。最多1请求、.10元估计结算阈值，connect10/read60秒，无重试。
2. 仅当步骤1真正成功并有合法usage/累计结算后，执行已保存scripts/run_a16_live_flash_pilot.py，新目录a28_sms_flash_pilot：SMS动态raw-history→checked-history，两路径同源码/full候选/实际划分707、808、909，最多各11监督fit含refit、3请求，600秒/worker60秒。两模型全封存后才读同一外层开发留出评分/新进程预测；正式test不打开。不拿旧版A20本地结果充当新版同源第三组因果对照。
3. 全批最多7请求、22监督fit，.70元为估计预留/usage结算停止阈值而非扣费硬上界。任意真实请求失败整批停止，不自动继续另一臂/重试；训练失败保留已有模型与账本。预计连通5–60秒、两路径合计2–8分钟；600秒单路径时限不是总耗时保证。
4. 若DNS失败，依据原ticket/run/hash核对是否未发出，可追加审计但不清空原始日志；其他失败不记零。保留原11历史未知、课程总费用未核验和最终预算报告不合格标记。

## 可复跑代码与完整命令

工作目录D:/codex/python_withai/runs/integration。两脚本已落盘并验收过离线/失败路径，内部protocol的A27/A16历史编号保留，本批目录与本文标识A28。复跑必须换从未存在的输出目录。

```powershell
& "D:\codex\python_withai\.venv\Scripts\python.exe" -m scripts.test_campus_flash_once --ledger "D:\codex\python_withai\runs\a23_real_local_history\ledger.sqlite" --api-key-file "D:\codex\python_withai\api_key.txt" --output "D:\codex\python_withai\runs\a28_flash_connectivity" --execute
```

仅成功后：

```powershell
& "D:\codex\python_withai\.venv\Scripts\python.exe" -m scripts.run_a16_live_flash_pilot --prepared "D:\codex\python_withai\runs\a07_prepared" --output "D:\codex\python_withai\runs\a28_sms_flash_pilot" --ledger "D:\codex\python_withai\runs\a23_real_local_history\ledger.sqlite" --api-key-file "D:\codex\python_withai\api_key.txt" --history-policy incremental-research --execute
```

保存实际请求数/token/课程计价、拟合数、耗时、动作与解释、两个导出模型的独立进程核验。只支持本小试与开发划分，不宣称未见迁移、统计显著性或完整六组C1完成。
