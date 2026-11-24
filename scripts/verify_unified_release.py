"""Publish and verify compact evidence from the canonical Agent and its Task2 follow-up."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from scripts.unified_released_study import EVIDENCE, PANELS, PLAN, ROOT, archive, predictions, read, require, sha, title_rows
from superagent.ab_run import sources


def publish(study, public):
    selections=['protocol.json','raw_inputs.json','scores.json','full_scores.json','calibration/weight.json','calibration/result.json',
                'calibration/predictions_frozen.json','calibration/development_predictions.csv','calibration/grouped_deltas.json',
                'validation/predictions_frozen.json','full_validation/predictions_frozen.json']
    for tag in PANELS:
        selections.extend([tag+'/model_frozen.json',tag+'/physical.jsonl'])
    for group,names in [('validation',['table','reference_table','titles','reference_titles']),('full_validation',['titles_full'])]:
        selections.extend(group+'/'+name+'_0/predictions.csv' for name in names)
    for name in selections:
        target=public/name
        target.parent.mkdir(parents=True,exist_ok=True)
        require(not target.exists() or sha(target)==sha(study/name),'Public evidence cannot change in place')
        shutil.copyfile(study/name,target)
    files=sorted(p for p in public.rglob('*') if p.is_file() and p.name!='SHA256SUMS')
    (public/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.relative_to(public).as_posix()+'\n' for p in files))


def verify(public, packages):
    listed={line[66:]:line[:64] for line in (public/'SHA256SUMS').read_text().splitlines()}
    actual={p.relative_to(public).as_posix():sha(p) for p in public.rglob('*') if p.is_file() and p.name!='SHA256SUMS'}
    require(listed==actual,'Public evidence hashes differ')
    protocol=read(public/'protocol.json')
    require(protocol['source']==sources() and protocol['plan']==sha(PLAN) and
            protocol['runner']==sha(ROOT/'scripts/unified_released_study.py') and
            protocol['reserve_ids']==sha(public/'title_reserve_ids.json'),'Published runtime, plan, or training split differs')
    for tag, facts in read(public/'raw_inputs.json').items():
        require(facts['training_features_identical'] and facts['training_labels_identical'] and facts['same_metric']
                and facts['training_rows']==PANELS[tag][0] and not any(
                part in n.lower() for n in facts['opened_source_members'] for part in ('_test','test.','_val','/val.')),
                'Original archive training interface differs from its packet')
    digest=hashlib.sha256(json.dumps(sources(),sort_keys=True).encode()).hexdigest()
    fits=0
    for tag in PANELS:
        record=read(public/tag/'model_frozen.json')
        workers=[json.loads(s) for s in (public/tag/'physical.jsonl').read_text().splitlines()]
        require(record['source_digest']==digest and record['invoked_as']=='superagent unified' and
                record['actions'][-1]['action']=='stop' and record['training_rows']==PANELS[tag][0] and
                record['new_api_calls']==0 and workers==record['fit_records'] and 1<=len(workers)==record['fits']<=10,
                'Native fit receipts, common source or budget disagree')
        fits+=len(workers)
    for score,group in [('scores.json','validation'),('full_scores.json','full_validation')]:
        results=read(public/score)
        seal=read(public/group/'predictions_frozen.json')
        for name,v in results.items():
            tag=name.removeprefix('reference_')
            file=public/group/(name+'_0')/'predictions.csv'
            require(sha(file)==v['prediction'] and seal[name]==[sha(file)]*2 and v['rows']==PANELS[tag][1]
                    and v['correct']/v['rows']==v['accuracy'],'Invalid frozen prediction or score')
            values=predictions(file,v['rows'])
            if name.startswith('reference_'):
                require(v['correct']==protocol['historical_expected_correct'][tag],'Reference score changed')
            if 'same_as_historical_rows' in v:
                other=predictions(public/group/('reference_'+tag+'_0')/'predictions.csv',v['rows'])
                require(int((values==other).sum())==v['same_as_historical_rows'] and
                        v['corrected']-v['lost']==v['correct']-results['reference_'+tag]['correct'],'Paired results disagree')
            if packages:
                package=archive(packages,tag)
                require(sha(package)==protocol['archives']['table' if tag=='table' else 'titles'],'Wrong course archive')
                if tag=='table':
                    with zipfile.ZipFile(package) as z:
                        truth=np.load(io.BytesIO(z.read('in_vehicle_coupon_recommendation/y_val.npy')),allow_pickle=False)
                else:
                    truth=np.array([r[2] for r in title_rows(package,'val',labels=True)])
                require(int((values==truth).sum())==v['correct'],'Raw course labels disagree')
    extra=read(public/'calibration/result.json')
    weights=read(public/'calibration/weight.json')
    require(weights['runner']==sha(ROOT/'scripts/unified_title_calibration.py') and weights['plan']==sha(PLAN) and
            weights['supervised_fits']==extra['supervised_fit_count']==1 and 0<=weights['weight']<=1 and
            extra['new_api_calls']==0 and not extra['designated_validation_used'],'Calibration provenance')
    grouped=read(public/'calibration/grouped_deltas.json')
    n=np.array(grouped['rows'])
    d=np.array(grouped['correct_difference'])
    require(n.sum()==extra['rows'] and d.sum()==extra['gained']-extra['lost']==extra['calibrated_correct']-extra['reference_correct'],
            'Unmatched calibration counts')
    rng=np.random.default_rng(28113)
    ratios=[]
    for _ in range(2000):
        idx=rng.integers(0,len(n),size=len(n))
        ratios.append(d[idx].sum()/n[idx].sum())
    ci=np.quantile(ratios,[.025,.975])
    require(np.allclose(ci,extra['paired_interval_fraction']) and bool(d.sum()>=5 and ci[0]>0)==extra['promoted'],'Paired interval/gate changed')
    pred=public/'calibration/development_predictions.csv'
    with pred.open() as f:
        rows=list(csv.DictReader(f))
    require(sha(pred)==extra['predictions'] and [r['row_id'] for r in rows]==list(map(str,range(extra['rows']))),
            'Training-only comparison changed')
    if packages:
        training=title_rows(archive(packages,'titles'),'train',labels=True)
        y=np.array([training[i][2] for i in read(public/'title_reserve_ids.json')])
        base=np.array([int(r['control']) for r in rows])==y
        challenger=np.array([int(r['calibrated']) for r in rows])==y
        require((int(base.sum()),int(challenger.sum()),int((challenger&~base).sum()),int((base&~challenger).sum()))==
                (extra['reference_correct'],extra['calibrated_correct'],extra['gained'],extra['lost']),
                'Original training reserve disagrees with calibration counts')
    print(f'Verified three autonomous runs, {fits} native fits, one separate calibration fit and zero model API calls.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publish',type=Path)
    parser.add_argument('--packages',type=Path)
    args=parser.parse_args()
    if args.publish:
        publish(args.publish,EVIDENCE)
    verify(EVIDENCE,args.packages)
