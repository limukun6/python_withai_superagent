"""Replay equal-weight probabilities and three-head majority using matched, already-fitted checkpoints."""
import argparse
import csv
import io
from pathlib import Path
import os
import sys
from unittest.mock import patch
import zipfile

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_local_component_pair import check, read, save
from superagent import ab_run
from superagent.ab_methods import TEXT_DEEP, TRANSFER_METHODS
from superagent.prediction_evidence import fingerprint


NAMES = {TEXT_DEEP:'deep', **{method:name for name,method in TRANSFER_METHODS.values()}}
SPLITS = ('primary','confirmation_1','confirmation_2')


def matched_runs(control,batch):
    records = {}
    for run in [control/'deep', *(batch/arm for arm in read(batch/'protocol.json')['order'])]:
        for trial,evidence in zip(read(run/'history.json'),read(run/'prediction_evidence.json'),strict=True):
            candidate = trial['candidate']
            if candidate not in NAMES or trial['status'] != 'completed':
                continue
            key = (trial['split_id'],NAMES[candidate])
            check((evidence['trial'],evidence['candidate_id'],evidence['split_id']) ==
                  (trial['trial'],candidate,key[0]), 'Training history misalignment')
            seal = dict(payload=evidence['evaluation_payload_sha256'],labels=evidence['label_sha256'],
                        predictions=evidence['prediction_sha256'],row_ids=evidence['row_ids_sha256']
                        if 'row_ids_sha256' in evidence else trial['row_ids_sha256'],params=trial['params'])
            if key in records:
                check(records[key][2] == seal,'Shared candidate gives different evidence in the same split')
            else:
                records[key]=(run,trial['trial'],seal)
    return records


def predictions(model,rows):
    check(type(model) is Pipeline and tuple(model.classes_) == (0,1),'Unexpected raw worker model')
    matrix = model[:-1].transform(rows)
    head = model[-1]
    prob = head.predict_proba(matrix)[:,1]
    hard = head.predict(matrix).astype(int)
    check(np.isfinite(prob).all() and np.all((prob>=0)&(prob<=1)),'Invalid probabilities')
    return hard,prob


def compare(new,base,y):
    a,b = new == y,base == y
    return dict(correct=int(a.sum()),reference_correct=int(b.sum()),gained=int((a&~b).sum()),
                lost=int((b&~a).sum()),rows=len(y))


def analysis(inputs, control, batch, out):
    out.mkdir(parents=True,exist_ok=True)
    records = matched_runs(control,batch)
    protocol = read(batch/'protocol.json')
    check(ab_run.sha(inputs/'features.zip') == protocol['identities']['packets']['features.zip'], 'Wrong reserve features')
    with zipfile.ZipFile(inputs/'features.zip') as z,z.open('rows.csv') as stream:
        rows=[(r['option_a'],r['option_b']) for r in csv.DictReader(io.TextIOWrapper(stream,encoding='utf-8-sig'))]
    outer={}
    detail={}
    sources={}
    for split in SPLITS:
        scores={}
        probs={}
        data=None
        for name in ('deep','e5','bge'):
            if (split,name) not in records:
                continue
            run,trial,seal=records[split,name]
            source=run/f'trial_{trial}.joblib'
            sources[split+'/'+name]=dict(trial=trial,source_model_sha256=ab_run.sha(source),seal=seal)
            checkpoint=joblib.load(source)
            check(ab_run.sha(run/(split+'.joblib')) == seal['payload'] == protocol['splits'][split]['evaluation_payload_sha256'],
                  'Different fitted/evaluation rows')
            if data is None:
                data=joblib.load(run/(split+'.joblib'))
            hard,p=predictions(checkpoint['model'],data['X_search'])
            check(fingerprint(data['y_search'])==seal['labels'] and fingerprint(hard)==seal['predictions']
                  and np.array_equal(hard,checkpoint['search_predictions']), 'Native predictions did not reproduce')
            scores[name],probs[name]=hard,p
            if split=='primary':
                outer[name]=predictions(checkpoint['model'],rows)[0]
        if data is None:
            continue
        y=np.asarray(data['y_search'])
        info=dict(rows=len(y),observed_correct={n:int((s==y).sum()) for n,s in scores.items()},pair_mean={})
        for variant in ('e5','bge'):
            if {'deep',variant} <= scores.keys():
                avg=(probs['deep']+probs[variant])/2
                first=np.array([l<r for l,r in data['X_search']])
                choice=((avg>.5)|((avg==.5)&first)).astype(int)
                info['pair_mean'][variant]=compare(choice,scores['deep'],y)
        if len(scores)==3:
            vote=(sum(scores.values())>=2).astype(int)
            info['three_votes']=compare(vote,scores['deep'],y)
        detail[split]=info
    frozen_save(out/'checkpoint_sources.json',sources)
    frozen_save(out/'inner.json',detail)
    check(len(outer)==3,'Three comparable primary checkpoints are required for the reserve test')
    outer['three_votes']=(sum(outer.values())>=2).astype(int)
    frozen_save(out/'predictions.json',{name:value.tolist() for name,value in outer.items()})
    frozen_save(out/'reserve_seal.json',dict(predictions=ab_run.sha(out/'predictions.json'),features=ab_run.sha(inputs/'features.zip'),
        sources=ab_run.sha(out/'checkpoint_sources.json'),protocol=ab_run.sha(ROOT/'docs/batches/B_three_encoder_agreement_1123_zh.md')))
    check(all(ab_run.sha(inputs/n)==protocol['identities']['packets'][n] for n in ('scoring_only.json','groups_only.json')),
          'Wrong reserve scoring data')
    y=np.array(read(inputs/'scoring_only.json')['labels'])
    groups=np.array(read(inputs/'groups_only.json'))
    check(len(y)==len(groups)==len(rows)==1693,'Wrong reserve row count')
    out_scores={name:compare(v,outer['deep'],y) for name,v in outer.items()}
    out_scores['three_votes']['paired']=paired((outer['three_votes']==y).astype(int)-(outer['deep']==y),groups)
    thresholds={}
    for variant in ('e5','bge'):
        comps=[v for v in detail.values() if variant in v['pair_mean']]
        wins=sum(v['pair_mean'][variant]['correct']>max(v['observed_correct'][n] for n in ('deep',variant)) for v in comps)
        totals={n:sum(v['observed_correct'][n] for v in comps) for n in ('deep',variant)}
        count=sum(v['rows'] for v in comps)
        extra=sum(v['pair_mean'][variant]['correct'] for v in comps)-max(totals.values()) if comps else 0
        thresholds[variant]=dict(matched_partitions=len(comps),partition_wins=wins,pooled_row_observations=count,
            pooled_delta_pp=100*extra/count if count else None,passed=bool(wins>=2 and count and extra/count>=.002))
    results=dict(reserve=out_scores,pair_average_gate=thresholds,inner_threeway_coverage=sum('three_votes' in x for x in detail.values()),
        prediction_seal=ab_run.sha(out/'reserve_seal.json'),new_fits=0,model_api=0,official_test=False)
    frozen_save(out/'results.json',results)
    print(results,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','inputs','control','batch','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    with patch.dict(os.environ,offline_environment(a.workspace.resolve())):
        analysis(*(getattr(a,n).resolve() for n in ('inputs','control','batch','output')))
