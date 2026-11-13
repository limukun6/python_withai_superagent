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
`--planner llm` with the endpoint, model and key path. Billing is currently
disabled: the client appends actual input/output Token counts, but does not
calculate or enforce any cost. It is
constrained to choose one untried local candidate;
it cannot request test data or arbitrary code execution.

## Design boundary

`run` is a development command.  It deliberately does not access a test
split.  A later final-evaluation command will be added only after the
selection policy and task contract are frozen.

