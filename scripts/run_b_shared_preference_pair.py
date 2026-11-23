"""Prepare or run the two native preference catalogs under a paired physical fit cap."""
import argparse
import csv
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.b_frozen_head_frontier import paired
from scripts.run_b_local_component_pair import check, command, read, save
from superagent import ab_run
from superagent.ab_methods import PREFERENCE_CATALOG_ENV, TEXT_DEEP, TEXT_EDIT
from superagent.cli import build_parser
from superagent.contracts import TaskSpec
from superagent.data import load_development_task
from superagent.metrics import MetricSpec


ARMS=('deep','deep-edit')
PLAN=REPO/'docs/batches/B_shared_preference_pair_protocol_1123_zh.md'
GUARD=REPO/'scripts/ab_acceptance_guard'
SEED=8123


def environment(workspace):
    return dict(os.environ,SUPERAGENT_MINILM_WEIGHTS=str(workspace/'work/pretrained_weights/all-MiniLM-L6-v2'),
        SUPERAGENT_DEEP_MINILM_WEIGHTS=str(workspace/'work/pretrained_weights/all-MiniLM-L12-v2'),
        HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',AB_B1_OFFLINE='1',
        PYTHONPATH=os.pathsep.join(map(str,(GUARD,REPO))),OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')


def interfaces(input_dir):
    return dict(production=ab_run.sources(),runner=ab_run.sha(Path(__file__)),plan=ab_run.sha(PLAN),
        inference_preflight=ab_run.sha(REPO/'research/b_accuracy_frontier_1123/native_delivery.json'),
        uncertainty=ab_run.sha(REPO/'scripts/b_frozen_head_frontier.py'),
        packet_helper=ab_run.sha(REPO/'scripts/run_b_local_component_pair.py'),
        physical_guard=ab_run.sha(GUARD/'sitecustomize.py'),
        files={name:ab_run.sha(input_dir/name) for name in (
            'anonymous.zip','features.zip','scoring_only.json','groups_only.json','protocol.json')})


def argv(inputs,out,arm):
    return ['run',str(inputs/'anonymous.zip'),'--portfolio','ab-v1','--scheduler','ab-v1','--planner','local',
            '--preference-catalog',arm,'--seed',str(SEED),'--fit-limit','11','--trial-timeout','300',
            '--time-limit','1200','--run-dir',str(out/arm)]


def prepare(inputs,out,env):
    out.mkdir(parents=True,exist_ok=False)
    prior=read(inputs/'protocol.json')
    frozen=interfaces(inputs)
    for name,key in [('anonymous.zip','training_sha'),('features.zip','feature_sha'),
                     ('scoring_only.json','locked_scores'),('groups_only.json','locked_groups')]:
        check(frozen['files'][name]==prior[key],'Input changed since original anonymous run')
    task=load_development_task(inputs/'anonymous.zip')
    spec=TaskSpec.from_task(task).record()
    check(prior['outer_title_groups_disjoint'] and len(task.y_train)==prior['rows']==6608 and
          prior['holdout']==1693 and task.X_val is None,'Wrong training scope')
    caps={}
    with patch.dict(os.environ,env),tempfile.TemporaryDirectory() as tmp:
        _,partitions=ab_run.freeze_partitions(task,Path(tmp),SEED,MetricSpec.for_task(task))
        for arm in ARMS:
            options=build_parser().parse_args(argv(inputs,out,arm))
            check(ab_run.validate_options(options)==11 and options.planner=='local','Wrong fit or planner limit')
            with patch.dict(os.environ,{PREFERENCE_CATALOG_ENV:arm}):
                caps[arm]=ab_run.registered_capabilities(spec,len(task.y_train),'full')
    check(TEXT_DEEP in caps['deep'] and TEXT_EDIT not in caps['deep'] and
          {k:v for k,v in caps['deep-edit'].items() if k!=TEXT_EDIT}==caps['deep']
          and list(caps['deep'])==[n for n in caps['deep-edit'] if n!=TEXT_EDIT],'Menus differ beyond the single candidate')
    save(out/'protocol.json',dict(interfaces=frozen,splits=partitions,seed=SEED,order=list(ARMS),
         allowed_physical_fits_per_arm=11,allowed_physical_fits_total=22,allowed_model_api=0,
         sizes={name:len(cap) for name,cap in caps.items()},additional_candidate=TEXT_EDIT,
         training_rows=prior['rows'],reserve_rows=prior['holdout'],official_test=False))


def execute(inputs,out,env):
    protocol=read(out/'protocol.json')
    check(protocol['interfaces']==interfaces(inputs),'Protocol inputs or implementation changed')
    save(out/'training_started.json',dict(process=os.getpid(),seed=SEED))
    receipts={}
    for arm in ARMS:
        physical=out/(arm+'.physical.jsonl')
        arm_env=dict(env,AB_B1_FIT_LOG=str(physical),AB_B1_PER_WORKER_FIT_LIMIT='1')
        rc,seconds=command(argv(inputs,out,arm),out/(arm+'.log'),arm_env,1260)
        counts=[json.loads(line) for line in physical.read_text().splitlines()] if physical.exists() else []
        run=out/arm
        if rc:
            save(out/(arm+'_failure.json'),dict(exit=rc,observed_fits=len(counts),seconds=seconds))
            raise RuntimeError(f'{arm} failed after {len(counts)} observed fits. See {arm}.log')
        outcome,account,manifest=[read(run/f) for f in ('run_outcome.json','classifier_fits.json','run_manifest.json')]
        n=len(counts)
        check(n==account['attempted_classifier_fits']==account['completed_classifier_fits']<=11,'Fit ledger mismatch')
        check(outcome['requested_planner_completed'] and outcome['full_refit_completed'] and outcome['new_api_requests']==0 and
              not (run/'llm_usage.jsonl').read_text().strip() and manifest['preference_catalog']==arm,'Run or usage mismatch')
        check(len(manifest['capabilities'])==protocol['sizes'][arm],'Run catalog mismatch')
        for key,value in protocol['splits'].items():
            actual=read(run/(key+'_split.json'))
            check(actual['row_ids_sha256']==value['row_ids_sha256'] and
                  ab_run.sha(run/(key+'.joblib'))==value['evaluation_payload_sha256'],'Split drift')
        history=read(run/'history.json')
        result=dict(actual_fits=n,seconds=seconds,candidate=outcome['selected_candidate'],model=ab_run.sha(run/'selected_model.joblib'),
            selection=ab_run.sha(run/'final_selection.json'),actions=ab_run.sha(run/'actions.json'),
            observed=[dict(candidate=r['candidate'],partition=r['split_id'],score=r['search_score']) for r in history],
            physical_worker_log=ab_run.sha(physical),model_api=0)
        save(out/(arm+'_receipt.json'),result)
        receipts[arm]=result
        print(arm,n,result['candidate'],flush=True)
    save(out/'models_frozen.json',receipts)
    exports={}
    for arm in ARMS:
        check(ab_run.sha(out/arm/'selected_model.joblib')==receipts[arm]['model'],'Frozen model changed')
        files=[]
        for index in range(2):
            d=out/f'{arm}_export_{index}'
            d.mkdir()
            target=d/'rows.csv'
            opts=['predict',str(inputs/'features.zip'),'--run-dir',str(out/arm),'--output',str(target)]
            rc,_=command(opts,d/'predict.log',env,600)
            check(rc==0 and not read(d/'prediction_manifest.json')['test_labels_read'],'Independent prediction failed')
            files.append(ab_run.sha(target))
        check(files[0]==files[1],'Independent exports differ')
        exports[arm]=files[0]
    save(out/'exports_frozen.json',dict(models=ab_run.sha(out/'models_frozen.json'),predictions=exports))
    check(protocol['interfaces']==interfaces(inputs),'Experiment inputs changed before scoring')
    y=np.array(read(inputs/'scoring_only.json')['labels'])
    groups=np.array(read(inputs/'groups_only.json'))
    hits={}
    for arm in ARMS:
        path=out/(arm+'_export_0')/'rows.csv'
        check(ab_run.sha(path)==exports[arm],'Unsealed export')
        with path.open() as f:
            rows=list(csv.DictReader(f))
        check(len(y)==len(groups)==1693 and [int(r['row_id']) for r in rows]==list(range(len(y))),'Unaligned score')
        hits[arm]=np.array([int(r['prediction']) for r in rows])==y
    a,b=[hits[arm] for arm in ARMS]
    save(out/'results.json',dict(rows=len(y),correct={arm:int(x.sum()) for arm,x in hits.items()},
        gained=int((b&~a).sum()),lost=int((a&~b).sum()),delta=paired(b.astype(int)-a,groups),
        actual_total_fits=sum(v['actual_fits'] for v in receipts.values()),model_api=0,new_api_rmb=0,
        development_seal=ab_run.sha(out/'exports_frozen.json'),course_validation_gate=bool(b.sum()>=1235 and b.sum()>a.sum()),
        official_test=False,H=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','execute'])
    p.add_argument('--workspace',type=Path,required=True)
    p.add_argument('--inputs',type=Path,default=REPO/'runs/local_components_02')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    (prepare if a.stage=='prepare' else execute)(a.inputs.resolve(),a.output.resolve(),environment(a.workspace.resolve()))
