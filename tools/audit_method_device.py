"""Verify the training-only decision study, replay masked decisions and compare matched fits."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import time
from unittest.mock import patch

import joblib
import numpy as np
from superagent.data import load_training_task
from superagent.method_choice import MINILM, NEURAL, SEMANTIC, select_action
from superagent.minilm_features import FILES as MINI_FILES
from superagent.models import predict_with_model
from superagent.pretrained_choice import digest
from superagent.tabular_encoder import level_names
from tools.pretrained_study import read, write, train_only
from tools.tabular_encoder_study import paired, usage


def matches_record(action,record):
    return action==(record['candidate'] if record['decision']=='try' else None)


def check_features(folder,trials,tokenizer):
    usage_file=folder/'downstream_usage.jsonl'
    events=[json.loads(v) for v in usage_file.read_text().splitlines()] if usage_file.exists() else []
    if not any(r['candidate']==MINILM for r in trials):
        assert not events
        return dict(tokens=0,forwards=0,weights_downloads=0)
    inner=joblib.load(folder/'inner_data.joblib')
    cfg=read(folder/f'method_{MINILM}.json')['params']
    names={s for i in cfg['columns'] for s in level_names(inner['X_train'][:,i])}
    keyset={hashlib.sha256(s.encode()).hexdigest() for s in names}
    db_path=(folder/'minilm_category_vectors.sqlite').resolve()
    with sqlite3.connect(f'file:{db_path}?mode=ro',uri=True) as db:
        vectors=dict(db.execute('select id,v from vectors'))
    assert keyset==set(vectors)
    matrix=np.stack([np.frombuffer(v,dtype='<f4') for v in vectors.values()])
    assert matrix.shape==(len(names),384) and np.allclose(np.linalg.norm(matrix,axis=1),1.,atol=2e-6)
    tokens=sum(len(item.ids) for item in tokenizer.encode_batch(sorted(names)))
    assert sum(r['input_tokens'] for r in events)==tokens and sum(r['sequences'] for r in events)==len(names)
    assert all(r['remote_calls']==0 and r['api_rmb']==0. and r['output_tokens']==0 for r in events)
    loads=[json.loads(s) for s in (folder/'model_loads.jsonl').read_text().splitlines()]
    return dict(tokens=tokens,forwards=len(events),names=len(names),dimensions=384,
                weights_downloads=sum('model.safetensors' in r['downloaded'] for r in loads if r.get('role')=='downstream_features'))


def compare_devices(folder):
    """Cross-over on one real inner training/search split, outside the search budgets."""
    import torch
    from superagent.scratch_classifier import ScratchClassifier
    inner=joblib.load(folder/'inner_data.joblib')
    decisions=read(folder/'trials.json')
    row=next((r for r in decisions if r['candidate']==NEURAL),None)
    if not row:
        return dict(completed=False,reason='First fold never executed the neural candidate.')
    gpu='cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else None)
    order=['cpu',gpu,gpu,'cpu'] if gpu else ['cpu','cpu']
    trials=[]
    for dev in order:
        record=dict(selected=dev,role='device diagnostic; same architecture and seed')
        with patch('superagent.scratch_classifier.device_probe',return_value=(dev,record)):
            start=time.perf_counter()
            model=ScratchClassifier(inner['kind'],row['params']).fit(inner['X_train'],inner['y_train'])
            elapsed=time.perf_counter()-start
        pred=predict_with_model(inner['kind'],model,inner['X_search'])
        trials.append(dict(device=dev,elapsed_seconds=elapsed,optimizer_seconds=model.training_metadata['optimizer_seconds'],
                           correct=int(np.sum(pred==inner['y_search'])),predictions=pred))
    reference=trials[0]['predictions']
    for entry in trials:
        pred=entry.pop('predictions')
        entry['paired_vs_first_cpu']=paired(reference,pred,inner['y_search'])
    medians={dev:float(np.median([r['elapsed_seconds'] for r in trials if r['device']==dev])) for dev in set(order)}
    selected=row['training_metadata']['device_decision']['selected']
    return dict(completed=True,rows=len(inner['y_search']),training_rows=len(inner['y_train']),order=order,trials=trials,
                median_wall_seconds=medians,chosen_by_live_run=selected,fastest_measured=min(medians,key=medians.get),
                scope='Four separate full fits in CPU/accelerator/accelerator/CPU order where an accelerator is available.')


def main():
    from tokenizers import Tokenizer
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','archive','ledger','minilm-weights'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device-check',action='store_true')
    a=p.parse_args()
    root=a.root.resolve()
    repo=Path(__file__).resolve().parents[1]
    protocol,study=read(root/'protocol.json'),read(root/'results.json')
    assert digest(a.archive)==protocol['archive_sha256']
    assert all(digest(repo/name)==sha for name,sha in protocol['source'].items())
    for name,(_,sha,size) in MINI_FILES.items():
        file=a.minilm_weights/name
        assert file.stat().st_size==size and digest(file)==sha
    tokenizer=Tokenizer.from_file(str(a.minilm_weights/'tokenizer.json'))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=256)
    opened=[]
    with train_only(a.archive,opened):whole=load_training_task(a.archive)
    seen=set()
    counts=dict(fits=0,final_models_reloaded=0,first_five_prediction_matches=0,replayed_actual_decisions=0,
                masked_paths_with_measured_outcomes=0,minilm_tokens=0,minilm_forwards=0,minilm_weight_downloads=0,
                qwen_tokens=0,qwen_forwards=0)
    folds=[]
    for i,split in enumerate(protocol['indices']):
        directory=root/f'fold_{i}'
        outer=read(directory/'outer_predictions.json')
        assert split['outer']==outer['rows'] and not seen.intersection(outer['rows'])
        seen.update(outer['rows'])
        x=whole.X_train[outer['rows']]
        y=whole.y_train[outer['rows']]
        np.testing.assert_array_equal(y,outer['truth'])
        log={}
        initial_preds=None
        for arm in ('classic','fixed','adaptive'):
            folder=directory/arm
            receipt=read(folder/'receipt.json')
            assert receipt['complete'] and digest(folder/'selected_model.joblib')==receipt['sha256']
            bundle=joblib.load(folder/'selected_model.joblib')
            assert receipt['winner']==bundle['candidate']
            pred=predict_with_model(whole.kind,bundle['model'],x)
            np.testing.assert_array_equal(pred,outer[arm])
            counts['final_models_reloaded']+=1
            counts['fits']+=receipt['fits']
            trials=read(folder/'trials.json')
            first=[joblib.load(folder/f'trial_{k}.joblib')['search_predictions'] for k in range(1,6)]
            if initial_preds is None:initial_preds=first
            else:
                for expected,actual in zip(initial_preds,first,strict=True):
                    np.testing.assert_array_equal(actual,expected)
                    counts['first_five_prediction_matches']+=1
            feature=check_features(folder,trials,tokenizer)
            counts['minilm_tokens']+=feature['tokens']
            counts['minilm_forwards']+=feature['forwards']
            counts['minilm_weight_downloads']+=feature['weights_downloads']
            qwen=usage(folder,a.ledger)
            counts['qwen_tokens']+=qwen['input_tokens']
            counts['qwen_forwards']+=qwen['forwards']
            log[arm]=[]
            for decision in receipt['decisions']:
                ev=decision['evidence']
                eligible=decision['eligible']
                used=decision['previous_methods']
                actual,_=select_action(ev,eligible,arm,used)
                assert matches_record(actual,decision)
                counts['replayed_actual_decisions']+=1
                masked,_=select_action(dict(ev,nonlinear_gain=0.,previous_delta=0.),eligible,'adaptive',used)
                log[arm].append(dict(actual=actual,without_score_deltas=masked,previous_methods=used,
                                     protected_incumbent=decision['protected_incumbent']))
        # Actual rows for the projected policy can be reused only if its entire path matches.
        masked_measured=all(t['actual']==t['without_score_deltas'] and t['actual'] is not None for t in log['fixed'])
        if masked_measured:counts['masked_paths_with_measured_outcomes']+=1
        folder=directory/'fixed'
        trials=read(folder/'trials.json')
        base=read(folder/'receipt.json')['incumbent']
        fits={'ordinary':base,'minilm':MINILM,'qwen':SEMANTIC}
        matched={}
        for label,method in fits.items():
            trial=next((r for r in trials if r['candidate']==method),None)
            if trial is None:continue
            if label!='ordinary':assert trial['params']['backbone']==base and trial['params']['backbone_params']==next(r['params'] for r in trials if r['candidate']==base)
            model=joblib.load(folder/f"trial_{trial['trial']}.joblib")['model']
            matched[label]=predict_with_model(whole.kind,model,x)
        folds.append(dict(fold=i+1,rows=len(y),decision_replays=log,masked_matches_fixed=masked_measured,
                          adaptive_vs_masked=paired(np.asarray(outer['fixed']),np.asarray(outer['adaptive']),y) if masked_measured else None,
                          matched_correct={k:int((v==y).sum()) for k,v in matched.items()},
                          minilm_vs_qwen=paired(matched['qwen'],matched['minilm'],y) if {'qwen','minilm'}<=matched.keys() else None))
    assert len(seen)==study['rows'] and counts['fits']==14*len(folds)*3
    summary=dict(counts=counts,folds=folds,training_members_opened=opened,api_generation_requests=0,actual_api_charge_rmb=0.,
                 qwen_conservative_rmb=study['course_rmb'],source_hashes_verified=True,
                 masked_policy='Keep initial five steps, gate, incumbent parameters and final confirmation. Project only nonlinear-minus-linear gain and previous-minus-incumbent score to zero at the two extra decisions.')
    if a.device_check:
        summary['device_check']=compare_devices(root/'fold_0'/'adaptive')
    write(root/'decision_and_device_audit.json',summary)
    print(counts,flush=True)
    if a.device_check:print(summary['device_check'],flush=True)


if __name__=='__main__':main()
