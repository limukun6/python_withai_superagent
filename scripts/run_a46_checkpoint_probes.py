"""Reuse already fitted checkpoints for recovery acceptance and tamper rejection; zero fits/HTTP."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil

import joblib

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from scripts import a38_predict_guard  # Forbid fitting and network in this process too.
from superagent.ab_run import sources
from superagent.data import load_training_task
from superagent.events import RunLogger
from superagent.partial_delivery import freeze_checkpoint


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','training','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);frozen=sources()
    manifest=read(args.run/'run_manifest.json')
    if sha(args.training)!=manifest['task_archive_sha256']:raise ValueError('Wrong training package')
    task=load_training_task(args.training);state=read(args.run/'state_3.json');caps=manifest['capabilities']
    trusted={}
    for split in ('primary','confirmation_1','confirmation_2'):
        payload=joblib.load(args.run/(split+'.joblib'));ids=read(args.run/(split+'_split.json'))['search_indices']
        trusted[split]=dict(row_ids=ids,labels=payload['y_search'].tolist(),evaluation_payload_sha256=sha(args.run/(split+'.joblib')))
    save(root/'protocol.json',dict(batch='A46-checkpoint-probes',source_files=frozen,training_source_files=manifest['source_files'],
        runner_sha256=sha(Path(__file__)),original_outcome_sha256=sha(args.run/'run_outcome.json'),
        classifier_fitting_forbidden=True,network_forbidden=True,external_validation_opened=False))
    results=[]
    for mode in ('valid','no_completed_primary','changed_score','wrong_split','incomplete_fit','changed_payload','changed_model'):
        directory=root/mode;directory.mkdir();active=deepcopy(state)
        for pattern in ('trial_*.joblib','trial_*.fits.json','primary.joblib'):
            for path in args.run.glob(pattern):shutil.copy2(path,directory/path.name)
        chosen=sorted(active['history'],key=lambda r:(-r['search_score'],r['candidate']))[0]
        if mode=='no_completed_primary':active.update(history=[],prediction_evidence=[])
        elif mode=='changed_score':active['history'][0]['search_score']=.999999
        elif mode=='wrong_split':active['prediction_evidence'][0]['split_id']='confirmation_1'
        elif mode=='incomplete_fit':
            ledger=directory/f'trial_{chosen["trial"]}.fits.json';value=read(ledger);value['completed_classifier_fits']=0;save(ledger,value)
        elif mode=='changed_payload':
            with (directory/'primary.joblib').open('ab') as file:file.write(b'changed payload')
        elif mode=='changed_model':
            path=directory/f'trial_{chosen["trial"]}.joblib';bundle=joblib.load(path)
            other=next(r for r in active['history'] if r['trial']!=chosen['trial'])
            bundle['model']=joblib.load(directory/f'trial_{other["trial"]}.joblib')['model'];joblib.dump(bundle,path)
        try:
            result=freeze_checkpoint(task,active,trusted,caps,directory,RunLogger(directory),'controlled pre-existing failure')
        except ValueError as error:
            if mode=='valid':raise
            if (directory/'final_selection.json').exists():raise ValueError('Rejected input produced a delivery')
            results.append(dict(case=mode,status='rejected',error=str(error)))
        else:
            if mode!='valid':raise ValueError('Corrupted evidence was accepted: '+mode)
            results.append(dict(case=mode,status='accepted_partial',recovery=result))
    save(root/'results.json',results)
    save(root/'completion.json',dict(status='passed',valid_partial_cases=1,tamper_or_missing_cases_rejected=6,
        actual_classifier_fits=0,real_api_requests=0,source_unchanged=sources()==frozen,original_run_modified=False))
    print(json.dumps(read(root/'completion.json')))
