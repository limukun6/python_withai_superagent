"""Probe reusable existing training statistics without changing any live planner input."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

from scripts.a_batch_common import save
from scripts.classifier_fit_observer import observe_classifier_fits
from scripts.run_a10_real_task_pilot import read,sha
from superagent.ab_run import sources
from superagent.data import load_training_task
from tools.a37_profile_contract import training_facts,append_common_profile,facts_sha256


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',type=Path,required=True)
    parser.add_argument('--readiness',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);frozen=sources()
    def offline(event,values):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise PermissionError('Network forbidden in profile prototype')
    sys.addaudithook(offline)
    entries={entry['task']:entry for entry in read(args.prepared/'readiness.json')}
    old={entry['task']:entry for entry in read(args.readiness/'readiness.json')}
    save(root/'protocol.json',dict(batch='A37-prototype',source_files=frozen,live_production_modified=False,
        runner_sha256=sha(Path(__file__)),prototype_sha256=sha(Path(__file__).resolve().parents[1]/'tools/a37_profile_contract.py'),
        scope='Train-only fact projection, no deployed client input change'))
    results=[]
    with observe_classifier_fits() as calls:
        for name in ('coupon','preference','sms','photo','bean'):
            entry=entries.get(name)
            source=Path(entry['training_archive']) if entry else Path(old[name]['source_archive'])
            expected=entry['training_archive_sha256'] if entry else old[name]['task_archive_sha256']
            if sha(source)!=expected:raise ValueError('Training package bytes changed')
            task=load_training_task(source);facts=training_facts(task)
            raw=dict(task_kind=task.kind,history=[]);checked={**raw,'checks':[],'memory':{}}
            first,second=append_common_profile(raw,checked,facts)
            assert first['training_profile']==second['training_profile']
            assert raw==dict(task_kind=task.kind,history=[])
            assert 'validation_label_counts' not in facts['statistics']
            before=facts_sha256(facts)
            renamed=SimpleNamespace(kind=task.kind,X_train=task.X_train,y_train=task.y_train,
                spec={**task.spec,'display_name':'anonymous renamed task','source_path':'renamed-input.zip'},
                name='anonymous renamed task')
            assert before==facts_sha256(training_facts(renamed))
            results.append(dict(task_for_report_only=name,training_archive_sha256=sha(source),facts=facts,
                facts_sha256=before,visible_json_utf8_bytes=len(json.dumps(facts,ensure_ascii=False).encode('utf-8')),
                raw_checked_common_profile_identical=True,display_name_and_path_invariant=True,deployed_in_AB=False))
    if calls:raise ValueError('A profile probe unexpectedly fitted a classifier')
    save(root/'results.json',results)
    save(root/'completion.json',dict(status='passed',tasks=len(results),actual_classifier_fits=0,real_api_requests=0,
        source_unchanged=sources()==frozen,production_profile_integration_performed=False))
    print(json.dumps(dict(status='passed',tasks=len(results),new_fits=0,new_API=0,source_unchanged=sources()==frozen)))
