"""Audit automatic output understanding without changing manifest inputs or reusing the wrong API receipt."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
from unittest.mock import patch

import numpy as np

from scripts.run_sa5_semantic_panel import read,save,sha
from superagent.data import load_training_task
from superagent.semantic_contract import resolve_training_task
from superagent.semantic_artifact import load_artifact
from tests.test_auto_semantics import standard_case
from tests.test_semantic_contract import Client
from tools.import_semantic_artifact import migrate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cases','live','run','ledger','output','private-output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);a.private_output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];before=sha(a.ledger)
    save(a.output/'protocol.json',dict(batch='SA16-auto-live-audit',runner_sha256=sha(Path(__file__)),
        live_receipt_sha256=sha(a.live/'completion.json'),baseline_sha256=sha(root/'research/sa16_local_baseline_01/completion.json'),
        maximum_classifier_fits=0,maximum_real_API_requests=0,prediction_answers_opened=False))
    controls=[]
    with tempfile.TemporaryDirectory() as temporary,patch.object(socket,'create_connection',side_effect=AssertionError('Control forbids networking')):
        for explicit in [False,True]:
            path,answer,expected=standard_case(Path(temporary)/f'{explicit}.zip',explicit);client=Client([answer])
            task=resolve_training_task(path,client,mode='auto',prediction_package=path)
            controls.append(dict(case='manifest' if explicit else 'plain_csv',complete=task.spec.get('output_recipe')==expected,
                                 simulated_reply_requests=client.request_count,real_API_requests=0))
    row=read(a.cases/'registry.json')[0];package=a.cases/row['package'];query=a.cases/row['query']
    before_task=load_training_task(package,read_declarations=True)
    after_task=load_artifact(a.run/'agent/semantic_artifact.json',package,query)
    same_input=(np.array_equal(before_task.X_train,after_task.X_train) and np.array_equal(before_task.y_train,after_task.y_train)
                and before_task.spec['input_schema']==after_task.spec['input_schema'] and before_task.spec['metric']==after_task.spec['metric'])
    local=resolve_training_task(package,mode='fallback',prediction_package=query)
    trace=read(a.live/'semantic_trace.json');auto=any(r['kind']=='semantic_mode_selected' and r.get('requested_mode')=='auto' and r.get('selected_mode')=='proactive' for r in trace)
    usage=read(a.live/'metered_usage.json');delta=read(a.live/'ledger_delta.json')
    linked=0<len(usage)==len(delta)<=3 and {r['budget_request_id'] for r in usage}=={r['id'] for r in delta}
    for record in delta:
        receipt=next(r for r in usage if r['budget_request_id']==record['id'])
        if record['status']!='known' or record['course_units']!=round(receipt['course_cost_rmb']*1e9):linked=False
    metadata=migrate(package,query,a.run/'agent/semantic_contract.json',a.ledger,a.private_output/'revalidated_artifact.json')
    semantic=[r for r in usage if r['role'].startswith('task_semantic')]
    correct_receipt=bool(semantic) and metadata['provider_provenance']['original_request_id']==semantic[-1]['budget_request_id']
    env=os.environ.copy()
    for key in list(env):
        if key.startswith(('SUPERAGENT_PLANNER_','SUPERAGENT_RESEARCH_','SUPERAGENT_ACCEPTED_')):env.pop(key)
    env.update(SUPERAGENT_OFFLINE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1');destination=a.private_output/'fresh/predictions.csv'
    proc=subprocess.run([sys.executable,'-m','unified_agent','predict',str(query.resolve()),'--run-dir',str(a.run.resolve()),
        '--output',str(destination.resolve())],cwd=root,env=env,stdin=subprocess.DEVNULL,capture_output=True,text=True,
        encoding='utf-8',errors='replace',timeout=90)
    (a.private_output/'predict.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
    identical=proc.returncode==0 and sha(destination)==sha(a.run/'delivery/predictions.csv')
    command=read(a.live/'invocation.json')['command'];defaults=not any(x in command for x in ('--task-interpretation','--task-analysis'))
    result=dict(status='passed' if all(r['complete'] for r in controls) and same_input and auto and linked and correct_receipt and identical and defaults and before==sha(a.ledger) else 'failed',
        simulated_controls=controls,real_auto_selection_observed=auto,default_human_CLI_flags_verified=defaults,
        authoritative_manifest_inputs_unchanged=same_input,legacy_fallback_has_output_recipe=bool(local.spec.get('output_recipe')),
        provider_usage_linked_to_canonical_ledger=linked,migration_uses_semantic_receipt=correct_receipt,
        fresh_prediction_identical=identical,ledger_unchanged=before==sha(a.ledger),
        real_live_roles=[{k:r.get(k) for k in ('role','input_tokens','output_tokens','course_cost_rmb')} for r in usage],
        actual_classifier_fits=0,new_API_requests=0,new_course_cost_rmb=0,quality_evaluation=False,
        scope='two simulated controls and one real development output contract; not independent blind task transfer')
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result,ensure_ascii=False))
    return result['status']!='passed'


if __name__=='__main__':raise SystemExit(main())
