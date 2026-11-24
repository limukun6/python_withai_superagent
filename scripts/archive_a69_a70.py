"""Compact D3 experiment evidence and actual resource reconciliation, with failures preserved."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.task_budget import TaskBudgetLedger


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('b1','feedback','ledger','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--recovery',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);index=[]
    def copy(source,target):
        if not source.exists():return
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        value=[json.loads(s) for s in source.read_text(encoding='utf-8').splitlines()] if source.suffix=='.jsonl' else read(source)
        index.append(dict(path=target.relative_to(a.output).as_posix(),original_runtime_bytes_sha256=sha(source),
            canonical_value_sha256=hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()))
    for name in ('protocol.json','completion.json','comparison.json','results.json','physical_fits.jsonl'):
        copy(a.b1/name,a.output/'A69'/name)
    for phase in ('prepared','answers','evaluation','behavior'):
        for name in ('protocol.json','completion.json','states.json','answers.json','comparison.json','results.json','physical_fits.jsonl'):
            copy(a.feedback/phase/name,a.output/'A70'/phase/name)
    for name in ('protocol.json','completion.json','window.json'):copy(a.feedback/name,a.output/'A70'/name)
    if a.recovery:
        for name in ('protocol.json','completion.json','comparison.json','results.json','physical_fits.jsonl','work_order.json'):
            copy(a.recovery/name,a.output/'A71'/name)
    for file in (a.feedback/'answers').glob('*/llm_usage.jsonl'):
        copy(file,a.output/'A70/answers'/file.parent.name/file.name)
    count=lambda root:sum(bool(line.strip()) for line in (root/'physical_fits.jsonl').read_text(encoding='utf-8').splitlines()) if (root/'physical_fits.jsonl').exists() else 0
    fits69=count(a.b1);fits70=count(a.feedback/'evaluation');fits71=count(a.recovery) if a.recovery else 0
    usage=[]
    for file in (a.feedback/'answers').glob('*/llm_usage.jsonl'):
        usage.extend(json.loads(line) for line in file.read_text(encoding='utf-8').splitlines() if line.strip())
    attempts=[r for r in usage if r['event']!='ledger_initialized'];calls=[r for r in attempts if r['event']=='llm_call']
    project=TaskBudgetLedger(a.ledger).project_summary()
    save(a.output/'resource_audit.json',dict(starting_tracked_A_fits_before_A67=1007,A67_reconstructed_regression_fits=15,A68_observed_fits=3,
        A69_observed_fit_starts=fits69,A70_observed_fit_starts=fits70,A71_observed_fit_starts=fits71,tracked_A_fits_after=1025+fits69+fits70+fits71,
        new_chat_attempt_records=len(attempts),successful_chat_responses=len(calls),
        input_tokens=sum(r['input_tokens'] for r in calls),output_tokens=sum(r['output_tokens'] for r in calls),
        known_new_course_cost_rmb=sum(r['course_cost_rmb'] for r in calls),project_usage_after=project,
        actual_platform_debit_rmb=None,course_budget_compliance='unverified_history',formal_test_or_H_opened=False))
    save(a.output/'index.json',dict(files=index,normalization='Git may normalize text to LF; use canonical JSON hashes to compare values.',
        models_datasets_raw_probabilities_credentials_or_sqlite_copied=False))
