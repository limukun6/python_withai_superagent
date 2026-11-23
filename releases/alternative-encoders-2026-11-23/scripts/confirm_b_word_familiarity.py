"""Confirm headline familiarity on title groups excluded from the exploratory evaluation sets."""
import argparse
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import zipfile

import joblib
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.b_frozen_head_frontier import paired
from scripts.check_b_word_surprisal import corpus_scores
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_local_component_pair import check,read
from superagent import ab_run
from superagent.ab_complement import canonical_probabilities
from superagent.ab_methods import TEXT_DEEP
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups


def column_data(packet,columns):
    with zipfile.ZipFile(packet) as archive,archive.open('rows.csv') as stream:
        return pd.read_csv(stream,usecols=columns,keep_default_na=False)


def run(inputs,old,native,old_pair,out):
    out.mkdir(parents=True,exist_ok=True)
    packet=inputs/'anonymous.zip'
    check(ab_run.sha(packet)==read(native/'protocol.json')['packets']['anonymous.zip'],'Wrong training input')
    frame=column_data(packet,['option_a','option_b'])
    pairs=list(frame[['option_a','option_b']].itertuples(index=False,name=None))
    groups=sample_groups(SimpleNamespace(kind='pair_text',X_train=pairs),'shared_text')
    sources={'8123':old/'deep-e5','18643':native/'autonomous'}
    seen=set()
    for folder in sources.values():
        for stage in ('primary','confirmation_1'):
            rows=read(folder/(stage+'_split.json'))['search_indices']
            seen.update(groups[rows])
    original_seen=len(seen)
    predictions={}
    summaries={}
    for seed,folder in sources.items():
        split=read(folder/'confirmation_2_split.json')
        train=split['train_indices']
        all_ids=np.asarray(split['search_indices'])
        keep=[idx for idx,row in enumerate(all_ids) if groups[row] not in seen]
        ids=all_ids[keep]
        check(len(ids)>0 and not seen.intersection(groups[ids]) and not set(groups[train]).intersection(groups[ids]),
              'Group overlap')
        seen.update(groups[ids])
        measure,corpus=corpus_scores(t for i in train for t in pairs[i])
        X=[pairs[i] for i in ids]
        gap=np.asarray([measure(a)-measure(b) for a,b in X])
        tie=np.array([a<b for a,b in X])
        pred=((gap<0)|((gap==0)&tie)).astype(int)
        extra=None
        # Only historical checkpoints on the very same registered row order are admissible.
        for directory in (folder,old_pair/'deep-edit') if seed=='8123' else (folder,):
            saved=read(directory/'confirmation_2_split.json')
            if (saved['row_ids_sha256'],saved['label_sha256'],saved['evaluation_payload_sha256'])!=(
                   split['row_ids_sha256'],split['label_sha256'],split['evaluation_payload_sha256']):
                continue
            row=next((r for r in read(directory/'history.json') if r['candidate']==TEXT_DEEP and r['split_id']=='confirmation_2'),None)
            if row is None:
                continue
            source=directory/f"trial_{row['trial']}.joblib"
            artifact=joblib.load(source)
            check(artifact['prediction_record']['label_sha256']==split['label_sha256'] and
                  artifact['prediction_record']['evaluation_payload_sha256']==split['evaluation_payload_sha256'],'Wrong anchor')
            hard=np.asarray(artifact['search_predictions'])[keep]
            pos=canonical_probabilities(artifact['model'],X)[1][:,1]
            mask=(abs(pos-.5)<=.05)&(abs(gap)>=.5)
            extra=dict(anchor=hard.tolist(),routed=np.where(mask,pred,hard).tolist(),eligible=int(mask.sum()),
                       checkpoint=ab_run.sha(source))
            break
        predictions[seed]=dict(ids=ids.tolist(),prediction=pred.tolist(),optional_anchor=extra)
        summaries[seed]=dict(rows=len(ids),groups=len(set(groups[ids])),ties=int((gap==0).sum()),
                             full_confirmation_rows=len(all_ids),corpus=corpus)
    frozen_save(out/'feature_only_predictions.json',predictions)
    frozen_save(out/'predictions_frozen.json',dict(predictions=ab_run.sha(out/'feature_only_predictions.json'),
        packet=ab_run.sha(packet),plan=ab_run.sha(ROOT/'docs/batches/B_familiarity_confirmation_1123_zh.md'),
        excluded_exploratory_groups=original_seen,group_assignment='first eligible confirmation: seed 8123, then seed 18643',
        partial=summaries))
    y=column_data(packet,['outcome'])['outcome'].to_numpy(dtype=int)
    grouped=[]
    correct=[]
    comparison=[]
    results={}
    for seed,pred in predictions.items():
        ids=np.array(pred['ids'])
        target=y[ids]
        split=read(sources[seed]/'confirmation_2_split.json')
        check(fingerprint(y[split['search_indices']])==split['label_sha256'],'Label scope changed')
        train_y=y[split['train_indices']]
        majority=int((train_y==1).sum()>(train_y==0).sum())
        hits=np.asarray(pred['prediction'])==target
        group=groups[ids]
        grouped.extend(group)
        correct.extend(hits)
        comparison.extend((target==majority).tolist())
        x=dict(**summaries[seed],correct=int(hits.sum()),accuracy=float(hits.mean()),majority_training_class=majority,
               majority_correct=int((target==majority).sum()),optional_anchor=None)
        if pred['optional_anchor']:
            ext=pred['optional_anchor']
            a=np.asarray(ext['anchor'])==target
            r=np.asarray(ext['routed'])==target
            x['optional_anchor']=dict(anchor_correct=int(a.sum()),routed_correct=int(r.sum()),eligible=ext['eligible'],
                 gained=int((r&~a).sum()),lost=int((a&~r).sum()))
        results[seed]=x
    h=np.asarray(correct,dtype=int)
    reference=np.asarray(comparison,dtype=int)
    group=np.asarray(grouped)
    check(sum(r['groups'] for r in results.values())==len(set(group)),'A group is counted twice')
    delta=paired(h-.5,group)
    gate=all(r['accuracy']>.5 for r in results.values()) and delta['group_bootstrap_95_pp'][0]>0
    result=dict(folds=results,pooled_rows=len(h),unique_groups=len(set(group)),correct=int(h.sum()),accuracy=float(h.mean()),
        majority_correct=int(reference.sum()),difference_from_chance=delta,difference_from_fold_training_majority=paired(h-reference,group),
        feature_signal_gate=bool(gate),new_supervised_fits=0,model_api=0,external_holdout_used=False,
        prediction_seal=ab_run.sha(out/'predictions_frozen.json'))
    frozen_save(out/'results.json',result)
    print({k:v for k,v in result.items() if k!='folds'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('workspace','inputs','old','native','old_pair','output'):
        p.add_argument('--'+n.replace('_','-'),dest=n,type=Path,required=True)
    a=p.parse_args()
    with patch.dict(os.environ,offline_environment(a.workspace.resolve())):
        run(*(getattr(a,n).resolve() for n in ('inputs','old','native','old_pair','output')))
