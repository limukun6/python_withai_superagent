"""Replay a menu gate on frozen original states; never invent counterfactual model scores."""
import argparse
import hashlib
import json
from pathlib import Path

from scripts.a_batch_common import save
from superagent.ab_run import sources
from superagent.candidate_coverage import project_routes


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    results=[]
    for task in ('coupon','preference'):
        for arm in ('R0','E0'):
            root=args.matrix/(task+'_'+arm)
            manifest=json.loads((root/'run_manifest.json').read_text(encoding='utf-8'))
            records=[json.loads(line) for line in (root/'events.jsonl').read_text(encoding='utf-8').splitlines()]
            requests=[row for row in records if row['kind']=='llm_request']
            proposals=[row for row in records if row['kind']=='ab_flash_proposal']
            for request,proposal in zip(requests,proposals,strict=True):
                context=json.loads(request['messages'][1]['content'])
                ids=tuple(row['id'] for row in context['available_candidates'])
                projected,audit=project_routes(dict(history=context['history'],available_candidates=context['available_candidates']),
                    manifest['capabilities'],ids)
                selected=proposal['proposal']['candidate']
                results.append(dict(task=task,arm=arm,trial=proposal['trial'],original_candidate=selected,
                    original_choice_allowed=selected in projected,projection=audit,
                    original_request_messages_sha256=hashlib.sha256(json.dumps(request['messages'],sort_keys=True).encode()).hexdigest()))
    save(args.output/'protocol.json',dict(batch='A31-replay',source_files=sources(),
        original_matrix_protocol_sha256=hashlib.sha256((args.matrix/'protocol.json').read_bytes()).hexdigest(),
        scope='Original states only; changed online histories and outcomes not simulated'))
    save(args.output/'results.json',results)
    completion=dict(status='completed',original_proposals=len(results),
        original_choices_outside_projected_menu=sum(not row['original_choice_allowed'] for row in results),
        actual_classifier_fits=0,real_api_requests=0,counterfactual_performance_evaluated=False)
    save(args.output/'completion.json',completion);print(json.dumps(completion))
