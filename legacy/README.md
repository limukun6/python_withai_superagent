# 个人分支统一之前的研发环境

根目录的 `superagent/`、`tests/` 和 `tools/` 采用 A/B 共同运行版本。之前个人分支的 Python 源码、测试、工具、依赖清单和说明已封存为 [hjl-848e52c-source.zip](hjl-848e52c-source.zip)。历史实验发布包、逐行预测及交接记录仍可从本分支的 `releases/` 和 `research/` 查看。

需要复现依赖原有目录布局的早期实验时，可以从同一仓库创建独立的历史检出。

```bash
git worktree add --detach ../superagent-hjl-history 848e52c9b7adc5fcf573239610e9d8709d1bd8b7
```
