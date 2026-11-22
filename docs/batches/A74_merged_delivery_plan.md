# A74：合并核心的真实执行与交付验收

起点9c7ceaf；先落本计划及runner，再执行。只有一条训练队列，不访问H或课程test、不调用API、不改代理、不自动重试。A起始累计1214次fit尝试。本批最多14次：匿名数值、明确对称、三类有序关系、已见SMS显式AUC各一条正式CLI路径，每条3fit含最终refit；另2次预先指定的新B无序哈希/有序交互worker动作性质验收。后两项不是自主效果实验。

四条路径均由同一入口自主选候选；保存实际候选菜单、状态、动作理由、fit计数和源码。每trial180秒、每path600秒、每path3fit，统一seed707。前三个为合成工程契约，只验接入与交付；SMS仅使用A47登记的training-only训练包，替换指标为roc_auc/positive_label=spam，不读外层标签、不报告新性能提升。其无标签预测样本从训练输入取前31行，仅作序列化一致性，不作泛化分数。

所有子进程禁网，worker外部观察fit，与正式账本核对。模型冻结后正式predict在另一个进程执行；再导出含确切源码的包，换cwd并python -I预测，CSV含AUC概率列须逐字节相同。推理过程禁止fit。明确对称模型交换预测及概率应保持；有序关系不施加翻转/对称假设。新B两动作经worker训练后检查对称概率或原生三类支持，保留model/fit账本本地，不提交二进制。

预计机器3–10分钟（硬上界约45分钟，遇首个失败停整批，保留完成项与失败计数）。先通过此批，再整合实验增量；不以软件测试代替实训、不把3fit短路径当11fit效果矩阵或H。source/runner/计划哈希、protocol/results/completion独占新目录，便于DeepSeek复跑。

```powershell
Set-Location 'D:\codex\python_withai\runs\b_review_20261122'
& 'D:\codex\python_withai\.venv\Scripts\python.exe' -m scripts.run_a74_merged_delivery --reference 'D:\codex\python_withai\runs\a47_catalog_Q' --output 'D:\codex\python_withai\runs\a74_merged_delivery'
```
