"""Verify published decisions, physical totals, submitted validation predictions and integration evidence."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'research/b_joint_opening_1123'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(test,message):
    if not test:
        raise ValueError(message)


def main():
    expected={'adaptive':5,'old_control':0,'coverage':6,'feedback':0,'missing_confirmation':1,'cold':10}
    for name, fits in expected.items():
        folder=DATA/name
        c=read(folder/'completion.json')
        require(c['status']=='complete' and c.get('result',c.get('results'))==sha(folder/'results.json')
                and c['physical_fits']==fits,'Unverified completion: '+name)
    study=DATA/'cold'
    r=read(study/'results.json')
    model=read(study/'model_frozen.json')
    d=read(study/'designated/results.json')
    seal=read(study/'designated/prediction_seal.json')
    predictions=study/'designated/predictions.csv'
    with predictions.open(encoding='utf-8',newline='') as f:
        c=csv.DictReader(f)
        require(c.fieldnames==['row_id','prediction'],'Unregistered prediction columns')
        rows=list(c)
    require([int(a['row_id']) for a in rows]==list(range(940)) and {a['prediction'] for a in rows}=={'0','1'}
            and seal['predictions']==sha(predictions) and d['prediction_seal']==sha(study/'designated/prediction_seal.json'),
            'Course validation predictions or order changed')
    require(d==r['designated'] and d['correct']==sum(d['confusion_true_by_predicted'][i][i] for i in (0,1))==689
            and d['rows']==940 and d['accuracy']==689/940 and d['frozen_model_training_rows']==6608
            and not d['course_test_member_opened'] and r['development_correct']==1237 and r['development_rows']==1693,
            'Scores or evaluation scope differ')
    require(r['all_replay_predictions_and_decisions_match'] and r['selected']==model['decision']['members']
            and model['actual_fits']==r['actual_new_fits']==10 and model['matched_prior_predictions']==model['actual_new_observations']==8
            and r['model_api_requests']==r['new_rmb']==0,'Unverified native decision or cost')
    standalone=read(study/'standalone_verified.json')
    integration=read(DATA/'integration.json')
    require(standalone['model']==seal['model']==model['model']==r['model']
            and standalone['exports']['designated']['prediction_sha256']==integration['exports']['designated']['predictions']==sha(predictions)
            and standalone['exports']['development']['prediction_sha256']==integration['exports']['development']['predictions']
            ==read(study/'exports_frozen.json')['development'],'Cross-process exports differ')
    require(integration['patch_applied_to_fresh_upstream'] and integration['tests_passed']==49
            and integration['patch_sha256']==sha(ROOT/'docs/integration/A117_B_joint_encoder.patch'),
            'A117 patch differs')
    print('Verified Task2 designated 689/940, grouped development 1237/1693, 10 cold fits, 22 stage fits, 0 model API calls and both standalone/A117 exports.')


if __name__=='__main__':
    main()
