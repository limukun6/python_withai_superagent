"""Export condition-blinded factual review inputs without checker labels or outcome scores."""
import argparse
import hashlib
from pathlib import Path
import random
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--answers',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[];mapping=[]
    for item in read(a.answers/'answers.json'):
        packet=read(Path(item['state_file']));state=packet['state'];caps=packet['capabilities']
        for mode in ('R','E'):
            identifier=hashlib.sha256((item['state_id']+'|'+mode).encode()).hexdigest()[:12]
            name=item[mode]['candidate'];proposal=item[mode]['proposal']['raw_proposal']
            fields=('trial','candidate','params','status','split_id','primary_metric','metric_spec','train_score','search_score',
                    'train_accuracy','search_accuracy','estimator')
            rows.append(dict(review_id=identifier,task_contract=state['task'],
                selected_registered_configuration={k:caps[name][k] for k in ('id','params','estimator')},
                history=[{k:r[k] for k in fields if k in r} for r in state['history']],proposal=proposal,
                independent_annotation=dict(factual_error=None,scope_or_reference_error=None,unsupported_causal_certainty=None,
                    appropriate_uncertainty=None,label_basis='',annotator='',checker_outputs_seen=None)))
            mapping.append(dict(review_id=identifier,state_id=item['state_id'],condition=mode))
    random.Random(713).shuffle(rows)
    save(a.output/'blinded_reviews.json',rows);save(a.output/'condition_key_keep_separate.json',mapping)
    save(a.output/'completion.json',dict(status='exported',reviews=len(rows),condition_hidden_in_review_file=True,
        machine_checker_labels_excluded=True,action_evaluation_scores_excluded=True,raw_training_examples_excluded=True,
        actual_classifier_fits=0,real_api_requests=0,answers_sha256=sha(a.answers/'answers.json')))
