"""Recheck real controlled replies, legacy safety, explicit annotation, artifact and ledger; zero new API/fit."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.run_sa5_semantic_panel import read,save,sha
from superagent.format_sources import package_files
from superagent.native_semantics import compile_native
from superagent.semantic_contract import compile_plan,evidence_workspace
from superagent.semantic_artifact import load_artifact,task_digest
from tools.import_semantic_artifact import migrate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['cases','live','private-run','ledger','output','private-output']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);a.private_output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];before=sha(a.ledger)
    save(a.output/'protocol.json',dict(batch='SA17-sealed-question-audit',runner_sha256=sha(Path(__file__)),
        live_receipt_sha256=sha(a.live/'completion.json'),maximum_classifier_fits=0,maximum_real_API_requests=0,
        scope='controlled interface outcomes and explicit post-hoc field annotation; not naturally occurring new task samples'))
    registry=read(a.cases/'registry.json');delta=read(a.live/'ledger_delta.json');requests=[];all_replies_match=True
    for row in registry:
        name=row['id'];usage=read(a.live/name/'usage.json');requests.extend([r for r in usage if r.get('event')=='llm_call'])
        events=[json.loads(s) for s in (a.private_run/name/'events.jsonl').read_text(encoding='utf-8').splitlines()]
        response=next(r for r in events if r['kind']=='llm_response')['content'].strip()
        if response.startswith('```'):response='\n'.join(response.splitlines()[1:-1])
        all_replies_match=all_replies_match and json.loads(response)==read(a.live/name/'model_reply.json')
    linked=len(requests)==len(delta)==3 and {r['budget_request_id'] for r in requests}=={r['id'] for r in delta}
    for row in delta:
        usage=next(r for r in requests if r['budget_request_id']==row['id'])
        linked=linked and row['status']=='known' and row['course_units']==round(usage['course_cost_rmb']*1e9)
    original=next(r['proposal'] for r in read(root/'research/sa15_default_live_01/semantic_trace.json') if r['kind']=='semantic_contract_rejected')
    row=registry[0];source=a.cases/row['package'];query=a.cases/row['query']
    legacy_blocked=False
    with package_files(source) as files,package_files(query) as prediction:
        workspace=evidence_workspace(files)
        try:compile_native(files,workspace,original,prediction_files=prediction)
        except ValueError:legacy_blocked=True
        annotated=deepcopy(original);annotated.update(version='semantic-native-v2',uncertainties=[
            dict(field='positive_label',question=original['uncertainties'][0],evidence_refs=['doc:README.md:0'])])
        converted=compile_native(files,workspace,annotated,prediction_files=prediction)
    restored=load_artifact(a.private_run/'U1/semantic_artifact.json',source,query)
    same_decoded=(np.array_equal(converted.X_train,restored.X_train) and np.array_equal(converted.y_train,restored.y_train)
                  and converted.spec['metric']==restored.spec['metric'] and converted.spec['output_recipe']==restored.spec['output_recipe'])
    meta=migrate(source,query,a.private_run/'U1/semantic_contract.json',a.ledger,a.private_output/'U1_revalidated_artifact.json')
    revalidated=load_artifact(a.private_output/'U1_revalidated_artifact.json',source,query)
    migrated=task_digest(revalidated)==task_digest(restored)
    # The target question must fail before any content read, even if the plan is null.
    row=registry[2];reply=read(a.live/'U3/model_reply.json');blocked_before_data=False
    with package_files(a.cases/row['package']) as files,package_files(a.cases/row['query']) as prediction:
        workspace=evidence_workspace(files)
        with patch.object(files,'read',side_effect=AssertionError('Core target question must precede data reads')) as guard:
            try:
                (compile_native if reply['version'].startswith('semantic-native') else compile_plan)(files,workspace,reply,prediction_files=prediction)
            except ValueError as error:blocked_before_data='target' in getattr(error,'uncertainty_fields',[])
            blocked_before_data=blocked_before_data and guard.call_count==0
    result=dict(status='passed' if all_replies_match and linked and legacy_blocked and same_decoded and migrated and blocked_before_data and before==sha(a.ledger) else 'failed',
        real_reply_files_match_transport_log=all_replies_match,real_usage_linked_to_canonical_ledger=linked,
        legacy_free_text_still_blocked=legacy_blocked,
        developer_annotated_old_question_accepted=True,annotation_is_post_hoc_not_original_model_output=True,
        decoded_input_goal_and_output_unchanged=same_decoded,real_typed_artifact_roundtrip_and_migration_match=migrated,
        core_target_blocked_before_data_read=blocked_before_data,ledger_unchanged=before==sha(a.ledger),
        real_live_input_tokens=sum(r['input_tokens'] for r in requests),real_live_output_tokens=sum(r['output_tokens'] for r in requests),
        real_live_course_cost_rmb=sum(r['course_cost_rmb'] for r in requests),
        actual_classifier_fits=0,new_API_requests=0,new_course_cost_rmb=0,
        natural_cost_saving_demonstrated=False,scope='three controlled question conditions, not 3 completed trained tasks')
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result,ensure_ascii=False))
    return result['status']!='passed'


if __name__=='__main__':raise SystemExit(main())
