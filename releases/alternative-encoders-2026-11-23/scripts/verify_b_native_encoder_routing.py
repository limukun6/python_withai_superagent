"""Verify the cold CLI path, prediction receipts, transplant, and research/engineering fit accounting."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ARCHIVE=ROOT/'research/b_native_encoder_routing_1123'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(value,message):
    if not value:
        raise ValueError(message)


def main(study):
    end=read(ARCHIVE/'completion.json')
    for k in ('protocol','model_frozen','exports','results'):
        require(sha(ARCHIVE/(k+'.json'))==end[k],k)
    plan=read(ARCHIVE/'protocol.json')
    require(plan['plan']==sha(ROOT/'docs/batches/B_native_encoder_routing_1123_zh.md') and
            plan['runner']==sha(ROOT/'scripts/run_b_native_encoder_routing.py'),'Plan or runner')
    model=read(ARCHIVE/'model_frozen.json')
    result=read(ARCHIVE/'results.json')
    physical=ARCHIVE/'physical.jsonl'
    require(sha(physical)==end['physical']==model['physical'] and len(physical.read_text().splitlines())==
            end['fits']==model['fit_count']==result['actual_fits']==plan['fit_cap']==10,'Physical training count')
    require(plan['study']==sha(study/'initial/completion.json') and plan['replication']==sha(study/'replication/completion.json'),
            'Study bindings')
    choice=read(study/'initial/decisions.json')['titles']['policy']
    require(all(model['routing'][k]==v for k,v in choice.items()) and choice['adopted'],'Actual decision')
    ex=read(ARCHIVE/'exports.json')
    a=read(ARCHIVE/'a117_receipt.json')
    original=read(study/'initial/results.json')['tasks']['titles']
    require(ex['prediction']==read(study/'initial/exports.json')['titles']['selected'] and ex['native']==ex['baseline']
            ==a['predictions'] and ex['native_repeats']==2 and ex['rows']==result['rows']==a['rows']==original['rows']
            and result['correct']==original['correct']['selected'],'Predictions or score')
    require(a['model']==model['model'] and a['actions']==result['actions']==9 and
            len(model['observations'])==result['identical_trial_records']==8 and a['first']==result['first']==choice['first']
            and a['threshold']==choice['threshold'],'Independent checkout')
    patches=read(ARCHIVE/'patch_chain.json')
    for target,v in patches.items():
        for patch in v['patches']:
            require(sha(ROOT/'docs/integration'/patch['file'])==patch['sha256'],target+' patch')
    for path,checksum in patches['B']['checked_sources'].items():
        if path.startswith('superagent/'):
            require(checksum==plan['sources'][path.removeprefix('superagent/')],'Source after patch')
    engineering=read(ARCHIVE/'engineering_checks.json')
    require(a['physical_fits']==a['model_api_calls']==result['model_api_calls']==result['new_rmb']==
            engineering['new_model_api_calls']==0 and not result['course_validation_or_test'],'Budget or evaluation set')
    print(json.dumps(dict(correct=result['correct'],rows=result['rows'],native_fits=end['fits'],
              additional_synthetic_fits=engineering['B']['additional_synthetic_classifier_fits'],
              a117_prediction_rows=a['rows'],new_model_api_calls=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study-dir',type=Path,default=ROOT/'research/b_encoder_cascade_1123')
    main(p.parse_args().study_dir)
