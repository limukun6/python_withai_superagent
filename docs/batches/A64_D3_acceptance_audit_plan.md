# A64：D3逐项验收与队友提交核对

用户授权继续补D3，并先写清验收清单、检查队友最新提交、判断D4必须修复项。本批只读Git/历史产物，重新执行已知无训练检查；0真实fit/0API，不打开H/test，不改变生产算法，不以复跑冒充新封存验证。

1. git fetch origin后记录各远端head、作者日期、已有bundle a787a28；缺新提交只代表当前远端无新上传，不代表队友本机无进展。不要直接合并旧分支全量代码。
2. 原计划逐项映射A1（96=48+48、8类）、A2（四任务12状态24请求/两新划分）、B1（4任务×2划分×2调度）、Q（五真实开发任务）、图像流程与交付。记录actual/evidence/gap/owner/acceptance，阶段编号不是验收证据。
3. 当前3a60de6生产源上回归A1语义9例、配置/引用19例、A05原36例、A06原12例；预期76检查条目，部分概念重叠，不称76独立缺陷，也不等同96预登记案例。A05原reserved已看过，只作回归。仅机制runner fixed/全量原案例，禁止调用会拟合的regression模式。A06模拟worker28次不计真实训练。
4. 用完成文件passed==cases与源码不变双重检查，不能只看子进程exit0。保留全部失败，失败不自动修案例/改期望。生成源/runner/结果SHA与completion。
5. 写D3清单与下一小批验收条件，区分必须补交付、已完成但负面结论、延后效果验证；原1000fit是建议线，本批不消耗，后续补实验要提前独立登记预算和版本。

可复跑：
```powershell
& 'D:\codex\python_withai\.venv\Scripts\python.exe' -m scripts.run_a64_d3_acceptance_audit --output 'D:\codex\python_withai\runs\a64_d3_acceptance_audit'
```

预计10–30秒，无训练/API；输出必须新目录。原日志、原封存身份与预算不改。
