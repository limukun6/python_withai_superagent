"""Re-evaluate the real previously rejected U2 reply under the corrected internal-AUC dependency."""
import argparse
import json
from pathlib import Path
import socket
from unittest.mock import patch

from scripts.run_sa5_semantic_panel import read,save,sha
from superagent.format_sources import package_files
from superagent.native_semantics import compile_native
from superagent.semantic_contract import evidence_workspace


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
    case=next(r for r in read(root/'runs/sa17_question_cases/registry.json') if r['id']=='U2')
    reply_path=root/'research/sa17_question_live_01/U2/model_reply.json';original_hash=sha(reply_path);reply=read(reply_path)
    save(a.output/'protocol.json',dict(batch='SA18-rejected-U2-continuation',original_reply_sha256=original_hash,
        corrected_rule_sha256=sha(root/'superagent/task_questions.py'),
        plan_sha256=sha(root/'docs/batches/SA18_RECOVERY_REALIGNMENT_20261127_zh.md'),
        maximum_classifier_fits=0,maximum_real_API_requests=0,scope='replay of an old real reply, not new autonomous model generation'))
    for key,hashed in [('package','package_sha256'),('query','query_sha256')]:
        if sha(root/'runs/sa17_question_cases'/case[key])!=case[hashed]:raise ValueError('Original controlled input changed')
    with patch.object(socket,'create_connection',side_effect=AssertionError('No new API')):
        with package_files(root/'runs/sa17_question_cases'/case['package']) as files,package_files(root/'runs/sa17_question_cases'/case['query']) as query:
            task=compile_native(files,evidence_workspace(files),reply,prediction_files=query)
    resolution=task.spec['uncertainty_resolution']
    result=dict(status='passed' if resolution['items'][0]['reason']=='binary_auc_invariant_under_swapping_named_class_probabilities' else 'failed',
        old_registered_expectation='blocked',current_contract_continues=True,metric=task.spec['metric'],
        training_rows=len(task.y_train),resolution=resolution,original_response_unchanged=original_hash==sha(reply_path),
        actual_classifier_fits=0,real_API_requests=0,course_cost_rmb=0,
        boundary='corrected one unnecessary rule; a complete cross-stage recovery controller is still pending')
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result,ensure_ascii=False));return result['status']!='passed'


if __name__=='__main__':raise SystemExit(main())
