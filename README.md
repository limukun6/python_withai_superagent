# General Algorithm SuperAgent

This repository contains a reproducible first version of the AIAA 3102
classification SuperAgent.  It detects supported task formats from their
schema, profiles only development data, runs a small autonomous candidate
portfolio, selects a model using validation accuracy, and records every
decision and metric.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Development run

The command below reads **only** `train` and `val`; it does not load test
features or test labels.

```powershell
python -m superagent run "D:\\学校\\python with ai\\Released_Task1-In_Vehicle_Coupon_Recommendation.zip"
python -m superagent run "D:\\学校\\python with ai\\Released_Task2-Headlines_CTR.zip"
```

Each command creates a timestamped directory below `runs/` containing the
task specification, data profile, candidate metrics, JSONL event log, zero
LLM-usage ledger, selected model and the fitted preprocessing pipeline.

## Candidate portfolios

Candidate selection depends on detected data schema rather than a task or
dataset name. Categorical tables use a majority baseline, regularised One-hot
logistic regression, ExtraTrees with two leaf sizes, and CatBoost with two
depth/regularisation settings. Pair-text inputs use a majority baseline plus
word and word/character TF-IDF difference classifiers with swap augmentation
and several regularisation strengths.

## Frozen-model prediction

Only after model selection is frozen, write test predictions without opening
test labels:

```powershell
python -m superagent predict "D:\\学校\\python with ai\\Released_Task1-In_Vehicle_Coupon_Recommendation.zip" `
  --run-dir runs\\categorical_tabular_YYYYMMDD_HHMMSS `
  --output final_predictions\\task1_predictions.csv
```

The output directory also receives `prediction_manifest.json`, which records
that no test labels were read or evaluated.

## Optional LLM planning

The default planner is local and costs nothing.  To make each next-experiment
decision with a permitted OpenAI-compatible Qwen or DeepSeek API, pass
`--planner llm` with the endpoint, model and key path. The client records
actual input/output Token counts, charges them using the configured RMB-per-1M
prices, and stops LLM planning before the default RMB 20 budget would be
exceeded. It is
constrained to choose one untried local candidate;
it cannot request test data or arbitrary code execution.

The current audited price configuration is in `superagent/pricing.py`, based
on the HPC AI Computing Platform table supplied on 2026-11-13:

| Model | Input RMB / 1M Tokens | Output RMB / 1M Tokens | Availability |
| --- | ---: | ---: | --- |
| DeepSeek-V4-Pro | 2.25 | 6.75 | enabled |
| DeepSeek-V4.1-Flash-1017 | 0.50 | 2.00 | enabled |
| Qwen3.8-27B-w8a8 | 0.10 | 0.34 | held until release-date evidence is recorded |

For example:

```powershell
python -m superagent run "TASK.zip" --planner llm `
  --api-url "https://aigc-api.hkust-gz.edu.cn/api/v2/chat/completions" `
  --api-key-file "D:\\path\\to\\api_key.txt" `
  --model DeepSeek-V4.1-Flash-1017
```

## Design boundary

`run` is a development command.  It deliberately does not access a test
split.  A later final-evaluation command will be added only after the
selection policy and task contract are frozen.

