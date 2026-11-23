"""Run the admitted text encoders through the standard offline planner on one frozen development split."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_local_component_pair import check, command, read, save
from scripts.run_b_shared_preference_pair import argv
from superagent import ab_run
from superagent.ab_methods import TRANSFER_METHODS


def identities(inputs, control, probe):
    return dict(modules=ab_run.sources(), scripts={name:ab_run.sha(ROOT/'scripts'/name) for name in (
        'run_b_alternative_encoders.py','probe_b_alternative_encoders.py','run_b_shared_preference_pair.py',
        'run_b_local_component_pair.py','b_frozen_head_frontier.py','ab_acceptance_guard/sitecustomize.py')},
        plan=ab_run.sha(ROOT/'docs/batches/B_alternative_encoders_protocol_1123_zh.md'),
        packets={name:ab_run.sha(inputs/name) for name in read(control/'protocol.json')['interfaces']['files']},
        control={name:ab_run.sha(control/name) for name in ('deep_receipt.json','exports_frozen.json',
            'deep_export_0/rows.csv','protocol.json','deep/history.json','deep/prediction_evidence.json')},
        probe=ab_run.sha(probe/'probe.json'))


def frozen_save(file, record):
    if file.exists():
        check(read(file) == record, 'Sealed data changed in ' + file.name)
    else:
        save(file,record)


def train_or_read(arm, inputs, control, batch, env, protocol):
    run = batch/arm
    physical = batch/(arm+'.physical.jsonl')
    if not run.exists() and not physical.exists():
        active_env = dict(env, AB_B1_FIT_LOG=str(physical), AB_B1_PER_WORKER_FIT_LIMIT='1')
        rc, _ = command(argv(inputs,batch,arm),batch/(arm+'.log'),active_env,1260)
        check(rc == 0, 'Standard entrypoint failed for '+arm)
    # Only completed runs may be resumed. A partially completed path is never retrained here.
    outcome, account, menu = [read(run/f) for f in ('run_outcome.json','classifier_fits.json','run_manifest.json')]
    count = len(physical.read_text().splitlines())
    check(count == account['attempted_classifier_fits'] == account['completed_classifier_fits'] <= 11, 'Actual fit mismatch')
    usage = [json.loads(line) for line in (run/'llm_usage.jsonl').read_text().splitlines()]
    check(len(usage) == 1 and usage[0]['event'] == 'ledger_initialized' and usage[0]['model'] is None
          and all(usage[0][key] == 0 for key in ('input_tokens','output_tokens','course_cost_rmb','actual_cost_rmb')),
          'Unexpected API usage')
    check(outcome['requested_planner_completed'] and outcome['full_refit_completed'] and outcome['source_unchanged']
          and not outcome['formal_test_accessed'] and outcome['new_api_requests'] == 0, 'Incomplete local delivery')
    check(menu['preference_catalog'] == arm and len(menu['capabilities']) == 34 and menu['seed'] == 8123, 'Menu or seed drift')
    reference = read(control/'deep/run_manifest.json')['capabilities']
    new = TRANSFER_METHODS[arm][1]
    check(new in menu['capabilities'] and [(k,v) for k,v in menu['capabilities'].items() if k != new]
          == list(reference.items()), 'Menu differs beyond the admitted method')
    for key, seal in protocol['splits'].items():
        split = read(run/(key+'_split.json'))
        check(split['row_ids_sha256'] == seal['row_ids_sha256'] and ab_run.sha(run/(key+'.joblib')) ==
              seal['evaluation_payload_sha256'], 'Split differs from control')
    history = read(run/'history.json')
    baseline = read(control/'deep/history.json')[:3]
    for candidate, common in zip(history[:3],baseline,strict=True):
        check(all(candidate[k] == common[k] for k in ('candidate','split_id','search_score','label_sha256',
              'row_ids_sha256','evaluation_payload_sha256')), 'Shared starting observations changed')
    evidence = read(run/'prediction_evidence.json')
    old_evidence = read(control/'deep/prediction_evidence.json')
    check([r['prediction_sha256'] for r in evidence[:3]] == [r['prediction_sha256'] for r in old_evidence[:3]], 'Shared predictions differ')
    receipt = dict(actual_fits=count,selected=outcome['selected_candidate'],model=ab_run.sha(run/'selected_model.joblib'),
        actions=ab_run.sha(run/'actions.json'),physical_log=ab_run.sha(physical),model_api=0,seconds=outcome['elapsed_seconds'],
        actual_trials=[dict(method=r['candidate'],split=r['split_id'],score=r['search_score'],status=r['status']) for r in history])
    frozen_save(batch/(arm+'_receipt.json'),receipt)
    return receipt


def export(arm, inputs, batch, env):
    seals = []
    for rep in range(2):
        out = batch/f'{arm}_export_{rep}'
        file = out/'rows.csv'
        if not out.exists():
            out.mkdir()
            rc,_ = command(['predict',str(inputs/'features.zip'),'--run-dir',str(batch/arm),'--output',str(file)],
                           out/'predict.log',env,600)
            check(rc == 0, 'Cannot export '+arm)
        check(not read(out/'prediction_manifest.json')['test_labels_read'], 'Prediction touched labels')
        seals.append(ab_run.sha(file))
    check(seals[0] == seals[1], 'Fresh-process exports differ')
    return seals[0]


def scores(file, expected, y):
    check(ab_run.sha(file) == expected, 'Predictions changed after freeze')
    with file.open(newline='') as f:
        rows=list(csv.DictReader(f))
    check([int(r['row_id']) for r in rows] == list(range(len(y))), 'Missing or displaced predictions')
    return np.array([int(r['prediction']) for r in rows]) == y


def main(workspace, inputs, control, probe, batch):
    env = offline_environment(workspace)
    batch.mkdir(parents=True,exist_ok=True)
    old = read(control/'protocol.json')
    fingerprint = identities(inputs,control,probe)
    preprocessing = read(probe/'probe.json')
    order = preprocessing['order']
    check(order in (['deep-e5'],['deep-e5','deep-bge']) and old['seed'] == 8123 and old['reserve_rows'] == 1693,
          'Unregistered scope or controls')
    check(old['interfaces']['files'] == fingerprint['packets'] and preprocessing['method_protocol_sha256'] == fingerprint['plan'],
          'A registered input or plan changed')
    protocol = dict(identities=fingerprint,order=order,seed=8123,training_rows=6608,reserve_rows=1693,splits=old['splits'],
        fit_cap_per_new_path=11,fit_cap_total=11*len(order),model_api_cap=0,formal_test_used=False,
        control_model=read(control/'deep_receipt.json')['model'])
    frozen_save(batch/'protocol.json',protocol)
    receipts={}
    for arm in order:
        receipts[arm]=train_or_read(arm,inputs,control,batch,env,protocol)
        print(arm,receipts[arm]['actual_fits'],receipts[arm]['selected'],flush=True)
    frozen_save(batch/'models_frozen.json',receipts)
    exports={'deep':read(control/'exports_frozen.json')['predictions']['deep']}
    for arm in order:
        check(ab_run.sha(batch/arm/'selected_model.joblib') == receipts[arm]['model'], 'Selected model changed')
        exports[arm]=export(arm,inputs,batch,env)
    frozen_save(batch/'exports_frozen.json',dict(models=ab_run.sha(batch/'models_frozen.json'),exports=exports))
    check(fingerprint == identities(inputs,control,probe), 'Protocol inputs or production code changed')
    y=np.asarray(read(inputs/'scoring_only.json')['labels'])
    groups=np.asarray(read(inputs/'groups_only.json'))
    check(len(y) == len(groups) == 1693,'Invalid reserve')
    baseline=scores(control/'deep_export_0/rows.csv',exports['deep'],y)
    check(baseline.sum() == 1224,'Reused control does not reproduce')
    result={}
    qualified=[]
    for arm in order:
        correct=scores(batch/(arm+'_export_0')/'rows.csv',exports[arm],y)
        chosen=receipts[arm]['selected'] == TRANSFER_METHODS[arm][1]
        if chosen and correct.sum() >= 1235:
            qualified.append(arm)
        result[arm]=dict(selected=receipts[arm]['selected'],selected_new=chosen,correct=int(correct.sum()),rows=len(y),
            gained=int((correct&~baseline).sum()),lost=int((baseline&~correct).sum()),delta=paired(
                correct.astype(int)-baseline,groups),actual_fits=receipts[arm]['actual_fits'])
    chosen=max(qualified,key=lambda arm:result[arm]['correct'],default=None)
    final=dict(control_correct=1224,results=result,course_validation_qualifier=chosen,model_api=0,
        actual_total_new_fits=sum(r['actual_fits'] for r in receipts.values()),control_fits_reused=read(control/'deep_receipt.json')['actual_fits'],
        export_seal_sha256=ab_run.sha(batch/'exports_frozen.json'),formal_test_used=False)
    frozen_save(batch/'results.json',final)
    print(json.dumps(final,ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('workspace','inputs','control','probe','output'):
        parser.add_argument('--'+arg,type=Path,required=True)
    args=parser.parse_args()
    main(*(getattr(args,key).resolve() for key in ('workspace','inputs','control','probe','output')))
