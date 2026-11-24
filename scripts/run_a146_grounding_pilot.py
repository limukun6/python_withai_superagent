"""At most twelve fixed-state Flash decisions; seal first, stop on any invalid response."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import time

from scripts.a_batch_common import hashes,save
from scripts.prepare_a146_grounding_pilot import ORDER
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a140_flash_projection import block_training
from tools.accepted_usage_snapshot import accepted_history

ROOT=Path(__file__).resolve().parents[1]
SCOPES={'coupon':'task1','sms':'sms_spam','flowers':'tensorflow_flowers_b_cohort'}
AUTH_SHA='a0f183573d10dbf46b33225f6ab29f2660176c3a6c8a7654705ddd3ee29679d5'


def ledger_rows(path):
    with sqlite3.connect(path) as db:
        return dict(requests=db.execute('SELECT * FROM requests ORDER BY id').fetchall(),
                    transitions=db.execute('SELECT * FROM transitions ORDER BY rowid').fetchall())


def unchanged_history(before,after):
    rows={r[0]:r for r in after['requests']}
    return (all(rows.get(r[0])==r for r in before['requests'])
            and after['transitions'][:len(before['transitions'])]==before['transitions'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('inputs','ledger','authorization','api-key-file','owner','output'):p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--validate-only',action='store_true');a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();source=hashes(ROOT)
    from superagent.task_budget import TaskBudgetLedger
    from superagent.research_budget import IncrementalResearchAccount,research_preflight
    from superagent.events import RunLogger
    from superagent.experiment_llm import ExperimentLLM
    from superagent.ab_flash import ENDPOINT,MODEL,validate_proposal
    from superagent.grounded_proposal import assess,GroundingError
    from superagent.explanation_audit import audit_explanation
    from scripts.campus_direct import campus_resolution,CAMPUS_IP
    block_training();ledger=TaskBudgetLedger(a.ledger);before=ledger.project_summary();rows_before=ledger_rows(a.ledger)
    save(out/'protocol.json',dict(batch='A146',source_files=source,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A146_flash_grounding_pilot_plan_20261124_zh.md'),
        inputs_completion_sha256=sha(a.inputs/'completion.json'),order=ORDER,scopes=SCOPES,
        authorization_sha256=sha(a.authorization),ledger_sha256_before=sha(a.ledger),project_before=before,
        starting_A_fits=1636,maximum_fits=0,maximum_requests=0 if a.validate_only else 12,
        estimated_stop_allowance_rmb=.60,automatic_retry=False,semantic_repair=False,validate_only=a.validate_only))
    answers=[]
    def owner(active):
        save(out/'current_ledger_owner.json',dict(owner='A',window_status='exclusive_active' if active else 'exclusive_idle',
            ledger_path=str(a.ledger.resolve()),ledger_sha256=sha(a.ledger),project_summary=ledger.project_summary(),
            accepted_history_sha256=AUTH_SHA,active_paid_queue=active,existing_unknowns_accepted_by_user=True,
            new_unknowns_require_stop=True))
    def seal(status,error=None):
        after=ledger.project_summary();old_preserved=unchanged_history(rows_before,ledger_rows(a.ledger));owner(False)
        result=dict(answers=answers,project_before=before,project_after=after,historical_rows_unchanged=old_preserved,
            known_incremental_cost_rmb=after['known_course_cost_rmb']-before['known_course_cost_rmb'],
            evidence_scope='One decision per cell; no next-action training or actual execution closure')
        save(out/'results.json',result)
        save(out/'completion.json',dict(status=status,error=error,actual_classifier_fits=0,
            real_API_requests=after['requests']-before['requests'],A_cumulative_fits_after=1636,
            historical_rows_unchanged=old_preserved,automatic_retry=False,source_unchanged=source==hashes(ROOT),
            elapsed_seconds=time.monotonic()-start,protocol_sha256=sha(out/'protocol.json'),results_sha256=sha(out/'results.json')))
    try:
        prior=read(a.owner)
        if prior['owner']!='A' or prior['active_paid_queue'] or prior['project_summary']!=before or prior['ledger_sha256']!=sha(a.ledger):
            raise ValueError('Unique idle A ledger owner does not match actual cumulative ledger')
        if sha(a.authorization)!=AUTH_SHA:raise ValueError('Accepted-history authorization differs')
        readiness=read(a.inputs/'completion.json');registration=read(a.inputs/'frozen_inputs.json')
        if (not readiness.get('eligible_for_A146') or read(a.inputs/'protocol.json')['source_files']!=source
                or sha(a.inputs/'frozen_inputs.json')!=readiness['frozen_inputs_sha256']):
            raise ValueError('Frozen inputs/source not eligible')
        for row in registration:
            folder=a.inputs/row['task']
            if sha(folder/'capabilities.json')!=row['capabilities_sha256']:raise ValueError('Catalogue changed')
            for condition,h in row['states'].items():
                if sha(folder/(condition+'_state.json'))!=h:raise ValueError('State changed')
            for context,h in row['contexts'].items():
                if sha(folder/(context+'.json'))!=h:raise ValueError('Request context changed')
            with accepted_history(a.authorization,AUTH_SHA):research_preflight(ledger,SCOPES[row['task']],.60)
        if not a.api_key_file.is_file():raise FileNotFoundError('Configured API key file missing')
        if a.validate_only:
            seal('validated');print(json.dumps(dict(status='validated',real_API_requests=0)));return
        private=out/'private';private.mkdir()
        with sqlite3.connect(a.ledger) as db,sqlite3.connect(private/'ledger_before.sqlite') as backup:db.backup(backup)
        owner(True)
        with accepted_history(a.authorization,AUTH_SHA),campus_resolution(CAMPUS_IP):
            for row in registration:
                task=row['task'];folder=a.inputs/task;caps=read(folder/'capabilities.json');scope=SCOPES[task]
                clients={mode:ExperimentLLM(ENDPOINT,str(a.api_key_file),MODEL,.60,max_requests=2,
                    allowed_served_models={MODEL},account=IncrementalResearchAccount(ledger,scope,str(out/task/mode)),streaming=True)
                    for mode in ('legacy','grounded')}
                for condition,mode in ORDER[task]:
                    if source!=hashes(ROOT):raise ValueError('Source changed during paid batch')
                    spent=ledger.project_summary()['known_course_cost_rmb']-before['known_course_cost_rmb']
                    remaining=.60-spent;research_preflight(ledger,scope,remaining)
                    state=read(folder/(condition+'_state.json'));ids=[c['id'] for c in state['available_candidates']]
                    context=condition+'_'+mode;target=out/task/context;target.mkdir(parents=True)
                    logger=RunLogger(target);logger.initialize_usage_ledger();client=clients[mode]
                    begun=time.monotonic()
                    answer=client.ask(read(folder/(context+'.json')),logger,'A146_'+context,
                                      cost_ceiling=client.spent+remaining)
                    save(target/'answer.json',answer)
                    usage=next(v for v in [json.loads(l) for l in logger.usage_path.read_text(encoding='utf-8').splitlines()] if v.get('event')=='llm_call')
                    receipt=dict(task=task,condition=condition,mode=mode,scope=scope,
                        context_sha256=sha(folder/(context+'.json')),answer_sha256=sha(target/'answer.json'),usage=usage,
                        elapsed_seconds=time.monotonic()-begun)
                    save(target/'sealed_answer.json',receipt);answers.append(receipt);save(out/'answers_sealed.json',answers)
                    if mode=='grounded':
                        try:proposal,report=assess(answer,state,caps,ids)
                        except GroundingError as e:
                            save(target/'grounding_report.json',e.report);raise
                        save(target/'grounding_report.json',report)
                    else:proposal=validate_proposal(answer,state,caps,ids,'calibrated-v2')
                    save(target/'contract.json',dict(valid=True,candidate=proposal.candidate))
                    audit=audit_explanation(answer,state['history'],{'id':proposal.candidate,'params':caps[proposal.candidate]['params']},
                        parameter_semantics=True,selected_facts={'classifier':caps[proposal.candidate]['estimator'],'source':'shared_ab_catalogue'})
                    save(target/'prose_audit.json',audit)
                    now=ledger.project_summary()
                    if now['requests']-before['requests']!=len(answers):raise ValueError('Unexpected concurrent ledger writer')
                    if not unchanged_history(rows_before,ledger_rows(a.ledger)):raise ValueError('Historical ledger rows changed')
                    research_preflight(ledger,scope,.60-(now['known_course_cost_rmb']-before['known_course_cost_rmb']));owner(True)
                    print(json.dumps(dict(task=task,condition=condition,mode=mode,status='sealed_valid',candidate=proposal.candidate,
                        input_tokens=usage['input_tokens'],output_tokens=usage['output_tokens'],cost_rmb=usage['course_cost_rmb'])),flush=True)
        if len(answers)!=12:raise ValueError('Incomplete registered pilot')
        seal('answers_sealed')
    except Exception as e:
        seal('stopped',type(e).__name__+': '+str(e));raise


if __name__=='__main__':main()
