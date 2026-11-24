"""At most nine real Flash proposals, one ledger, no retries and no supervised fits."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
import sys
import time

from scripts.a_batch_common import save,hashes
from scripts.run_a10_real_task_pilot import read,sha
from tools.accepted_usage_snapshot import accepted_history

ROOT=Path(__file__).resolve().parents[1]
SCOPES={'coupon':'task1','sms':'sms_spam','flowers':'tensorflow_flowers_b_cohort'}
ORDER={'coupon':['raw','full','compact'],'sms':['full','compact','raw'],'flowers':['compact','raw','full']}


def block_training():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.svm import SVC,LinearSVC
    from sklearn.dummy import DummyClassifier
    def forbidden(*a,**kw):raise RuntimeError('Supervised fits forbidden in A140')
    for cls in (LogisticRegression,ExtraTreesClassifier,SVC,LinearSVC,DummyClassifier):cls.fit=forbidden
    try:
        from catboost import CatBoostClassifier
        CatBoostClassifier.fit=forbidden
    except ImportError:pass


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('inputs','ledger','authorization','api-key-file','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();source=hashes(ROOT)
    from superagent.task_budget import TaskBudgetLedger
    from superagent.research_budget import research_preflight,IncrementalResearchAccount
    from superagent.events import RunLogger
    from superagent.experiment_llm import ExperimentLLM
    from superagent.ab_flash import ENDPOINT,MODEL,validate_proposal
    from scripts.campus_direct import campus_resolution,CAMPUS_IP
    block_training();ledger=TaskBudgetLedger(a.ledger);before=ledger.project_summary();auth_sha=sha(a.authorization)
    save(out/'protocol.json',dict(batch='A140',source_files=source,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A140_flash_projection_protocol_20261124_zh.md'),
        inputs_completion_sha256=sha(a.inputs/'completion.json'),order=ORDER,scopes=SCOPES,
        authorization_sha256=auth_sha,ledger_sha256_before=sha(a.ledger),project_before=before,
        starting_A_fits=1614,maximum_fits=0,maximum_requests=0 if a.validate_only else 9,
        total_estimated_stop_allowance_rmb=.60,automatic_retry=False,validate_only=a.validate_only))
    answers=[]
    def owner(active):
        value=dict(owner='A',window_status='exclusive_active' if active else 'exclusive_idle',
                   ledger_path=str(a.ledger.resolve()),ledger_sha256=sha(a.ledger),project_summary=ledger.project_summary(),
                   accepted_history_sha256=auth_sha,active_paid_queue=active,existing_unknowns_accepted_by_user=True,
                   new_unknowns_require_stop=True)
        save(out/'current_ledger_owner.json',value)
    try:
        readiness=read(a.inputs/'completion.json');registration=read(a.inputs/'frozen_inputs.json')
        if not readiness.get('eligible_for_A140') or read(a.inputs/'protocol.json')['source_files']!=source:
            raise ValueError('Projection/source not eligible for the registered paid trial')
        for r in registration:
            folder=a.inputs/r['task']
            if sha(folder/'state.json')!=r['state_sha256'] or sha(folder/'capabilities.json')!=r['capabilities_sha256']:
                raise ValueError('Frozen state/capabilities changed')
            for view,h in r['views_sha256'].items():
                if sha(folder/(view+'.json'))!=h:raise ValueError('Frozen request view changed')
            with accepted_history(a.authorization,auth_sha):research_preflight(ledger,SCOPES[r['task']],.20)
        if not a.api_key_file.is_file():raise FileNotFoundError('Configured local API key is missing')
        if a.validate_only:
            save(out/'completion.json',dict(status='validated',actual_classifier_fits=0,real_API_requests=0,
                 project_after=ledger.project_summary(),elapsed_seconds=time.monotonic()-start));return
        private=out/'private';private.mkdir()
        with sqlite3.connect(a.ledger) as db,sqlite3.connect(private/'ledger_before.sqlite') as backup:db.backup(backup)
        owner(True)
        with accepted_history(a.authorization,auth_sha),campus_resolution(CAMPUS_IP):
            for r in registration:
                task=r['task'];folder=a.inputs/task;state=read(folder/'state.json');caps=read(folder/'capabilities.json')
                ids=[c['id'] for c in state['available_candidates']];scope=SCOPES[task]
                account=IncrementalResearchAccount(ledger,scope,str(out/task))
                client=ExperimentLLM(ENDPOINT,str(a.api_key_file),MODEL,.20,max_requests=3,
                                     allowed_served_models={MODEL},account=account,streaming=True)
                for view in ORDER[task]:
                    if hashes(ROOT)!=source:raise ValueError('Source changed during paid batch')
                    research_preflight(ledger,scope,.20)
                    target=out/task/view;target.mkdir(parents=True)
                    logger=RunLogger(target);logger.initialize_usage_ledger();payload=read(folder/(view+'.json'))
                    begun=time.monotonic();answer=client.ask(payload,logger,'A140_'+view)
                    save(target/'answer.json',answer)
                    usage=[json.loads(l) for l in logger.usage_path.read_text(encoding='utf-8').splitlines()]
                    usage=next(v for v in usage if v.get('event')=='llm_call')
                    receipt=dict(task=task,view=view,scope=scope,context_sha256=sha(folder/(view+'.json')),
                         answer_sha256=sha(target/'answer.json'),usage=usage,elapsed_seconds=time.monotonic()-begun)
                    save(target/'sealed_answer.json',receipt);answers.append(receipt);save(out/'answers_sealed.json',answers)
                    # Contract failure stops the queue, never calls again to repair a response.
                    proposal=validate_proposal(answer,state,caps,ids,'calibrated-v2')
                    save(target/'contract.json',dict(valid=True,candidate=proposal.candidate))
                    if ledger.project_summary()['requests']-before['requests']!=len(answers):
                        raise ValueError('Unique ledger count differs from this exclusive queue')
                    research_preflight(ledger,scope,.20);owner(True)
                    print(json.dumps(dict(task=task,view=view,status='sealed',input_tokens=usage['input_tokens'],
                         output_tokens=usage['output_tokens'],cost_rmb=usage['course_cost_rmb'])),flush=True)
        after=ledger.project_summary();owner(False)
        if len(answers)!=9 or source!=hashes(ROOT):raise ValueError('Incomplete nine-view trial or source drift')
        save(out/'results.json',dict(answers=answers,project_before=before,project_after=after,
             known_incremental_cost_rmb=after['known_course_cost_rmb']-before['known_course_cost_rmb'],semantic_evaluation_pending=True))
        save(out/'completion.json',dict(status='answers_sealed',actual_classifier_fits=0,
             real_API_requests=after['requests']-before['requests'],A_cumulative_fits_after=1614,
             elapsed_seconds=time.monotonic()-start,source_unchanged=True,
             protocol_sha256=sha(out/'protocol.json'),results_sha256=sha(out/'results.json')))
    except Exception as e:
        after=ledger.project_summary();owner(False)
        save(out/'results.json',dict(answers=answers,project_before=before,project_after=after,
             known_incremental_cost_rmb=after['known_course_cost_rmb']-before['known_course_cost_rmb']))
        save(out/'completion.json',dict(status='stopped',error=str(e),actual_classifier_fits=0,
             real_API_requests=after['requests']-before['requests'],A_cumulative_fits_after=1614,
             automatic_retry=False,elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':main()
