"""One explicit Flash connectivity test, no training, no retries, same client and real ledger.

Default preflight never reads the API key or sends a request. --execute opts in.
"""
import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

from scripts.a_batch_common import save
from superagent.ab_flash import preflight, MODEL, ENDPOINT
from superagent.ab_run import sources
from superagent.events import RunLogger
from superagent.experiment_llm import ExperimentLLM
from superagent.research_budget import IncrementalResearchAccount


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    client = None
    ledger = None
    before = None
    save(root / 'protocol.json', dict(batch='A27-single-request', source_files=sources(),
        execute_requested=args.execute, maximum_requests=1, maximum_classifier_fits=0,
        endpoint=ENDPOINT, model=MODEL, connect_timeout_seconds=10, read_timeout_seconds=60,
        task_budget_id=args.task_budget_id, purpose='Connectivity probe charged to the next SMS planner batch',
        streaming=True, trust_env=False, automatic_retries=False, estimate_allowance_rmb=.10,
        history_policy='incremental-research', course_budget_compliance='unverified_history'))
    try:
        options = SimpleNamespace(api_url=None, model=None, api_key_file=args.api_key_file,
            budget_ledger=args.ledger, task_budget_id=args.task_budget_id, llm_budget_rmb=.10,
            history_policy='incremental-research')
        ledger, allowance, before = preflight(options)
        save(root / 'budget_before.json', dict(task=before, project=ledger.project_summary()))
        if not args.execute:
            value = dict(status='validated_not_executed', request_attempts=0, classifier_fits=0, key_read=False)
        else:
            logger = RunLogger(root)
            logger.initialize_usage_ledger()
            account = IncrementalResearchAccount(ledger, args.task_budget_id, str(root))
            client = ExperimentLLM(ENDPOINT, str(args.api_key_file), MODEL, allowance,
                max_requests=1, allowed_served_models={MODEL}, account=account, streaming=True)
            answer = client.ask(dict(instruction='Return exactly the JSON object {"ok":true}.'), logger, 'connectivity')
            save(root / 'answer.json', answer)
            if answer.get('ok') is not True:
                raise ValueError('Model responded, but did not return the requested ok:true object')
            value = dict(status='passed', request_attempts=client.request_count, successful_json_response=True,
                classifier_fits=0, known_new_course_cost_rmb=client.spent, source_unchanged=sources() == read_sources(root))
    except Exception as error:
        value = dict(status='failed', error=str(error), request_attempts=client.request_count if client else 0,
            classifier_fits=0, automatic_retry=False)
    value.update(elapsed_seconds=time.monotonic() - start, course_budget_compliance='unverified_history',
        valid_for_budgeted_final_report=False, project_usage_after=ledger.project_summary() if ledger else None)
    save(root / 'completion.json', value)
    print(json.dumps(value))
    return 1 if value['status'] == 'failed' else 0


def read_sources(root):
    return json.loads((root / 'protocol.json').read_text(encoding='utf-8'))['source_files']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--api-key-file', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--task-budget-id', default='sms_spam')
    parser.add_argument('--execute', action='store_true')
    raise SystemExit(run(parser.parse_args()))
