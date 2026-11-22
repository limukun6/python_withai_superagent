# A63：AUC故障恢复及源码自包含交付，零训练

输入仅A62已成功模型、原training-only包和概率证据。禁止fit/网络，复制到新目录后注入故障，不修改原始成功路径。预算0fit/0API，A累计仍999。

先验收合法partial；再分别移除score_record、修改矩阵、交换类名但不交换概率、修改metric、label绑定，以及将模型替换成“硬预测完全相同、概率相反”的受控包装器。六类损坏都应在生成最终交付前拒绝；不能只检查硬标签。反例是故障注入，不是自然错误发生率。

合法partial与完整A62模型均导出包含确切项目源码的预测包，`python -I`从另一cwd执行，校验源包来自bundle、预测CSV概率字节一致；无需API、训练数据或原仓库导入。沿用现有依赖环境，若可用再用A57独立venv复核完整包，不重建/安装环境。partial验证只证明已拟合模型可用，保留未完整重训/partial标记。

```powershell
& 'D:\codex\python_withai\.venv\Scripts\python.exe' -m scripts.run_a63_auc_recovery_delivery --reference 'D:\codex\python_withai\runs\a62_auc_official_loop' --output 'D:\codex\python_withai\runs\a63_auc_recovery_delivery'
```

预计5–30秒，任何失败保留原错误，不重试训练。所有新输出目录独占创建。报告与A62分开，不能把故障注入当新的真实任务/Flash实验。
