"""Post-selection, zero-refit diagnostic for the same backbone with three feature maps."""
import argparse
from pathlib import Path

import joblib
import numpy as np
from superagent.data import load_training_task
from superagent.models import predict_with_model
from superagent.pretrained_choice import digest
from superagent.tabular_encoder import PERMUTED, SEMANTIC, MISSING, level_names
from .pretrained_study import read, write, train_only
from .tabular_encoder_study import paired


def evaluate_fold(folder, whole):
    scores = read(folder/'outer_predictions.json')
    ids = np.asarray(scores['rows'])
    X, y = whole.X_train[ids], whole.y_train[ids]
    np.testing.assert_array_equal(y,scores['truth'])
    gates = [read(folder/a/'pretrained_decision.json') for a in ('semantic','permuted')]
    if any(g['decision']!='try' for g in gates):
        return dict(rows=len(y),available=False,reason='One or more fifth-slot policies skipped the transfer candidate.')
    real, shuffled = gates
    params = real['params']
    if ({k:v for k,v in params.items() if k!='assignment'}
            != {k:v for k,v in shuffled['params'].items() if k!='assignment'}):
        raise ValueError('Different encoder configuration or backbone')
    split = read(folder/'semantic'/'split.json')
    if split!=read(folder/'permuted'/'split.json'):
        raise ValueError('Candidate training rows differ')
    search = joblib.load(folder/'semantic'/'inner_data.joblib')
    trials = read(folder/'semantic'/'trials.json')
    incumbent = next(r for r in trials[:4] if r['candidate']==real['protected_incumbent'])
    refs = {'original':('semantic',incumbent['trial'],incumbent['candidate']),
            'semantic':('semantic',5,SEMANTIC),'permuted':('permuted',5,PERMUTED)}
    predictions, hashes = {}, {}
    for name, (arm,index,candidate) in refs.items():
        path = folder/arm/f'trial_{index}.joblib'
        artifact = joblib.load(path)
        if artifact['metrics']['resolved_configuration']['id']!=candidate:
            raise ValueError('Candidate artifact changed')
        model = artifact['model']
        np.testing.assert_array_equal(artifact['search_predictions'],predict_with_model(whole.kind,model,search['X_search']))
        predictions[name] = predict_with_model(whole.kind,model,X)
        hashes[name] = digest(path)
    low = np.zeros(len(y), dtype=bool)
    for col in params['columns']:
        counts = level_names(search['X_train'][:,col])
        median, maximum = np.median(list(counts.values())), max(counts.values())
        for i, value in enumerate(X[:,col]):
            name = str(value).strip()
            if name.casefold() not in MISSING and (name not in counts or (counts[name]<=median and counts[name]<maximum)):
                low[i] = True
    def compare(mask):
        subset = {a:values[mask] for a,values in predictions.items()}
        return dict(rows=int(mask.sum()),correct={a:int(np.sum(values==y[mask])) for a,values in subset.items()},
                    semantic_vs_original=paired(subset['original'],subset['semantic'],y[mask]),
                    semantic_vs_permuted=paired(subset['permuted'],subset['semantic'],y[mask]))
    return dict(rows=len(y),available=True,backbone=real['protected_incumbent'],fitting_rows=len(search['y_train']),
                models_sha256=hashes,all=compare(np.ones(len(y),dtype=bool)),lower_support=compare(low),other=compare(~low))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task1',type=Path,required=True)
    p.add_argument('--first-stage-root',type=Path,required=True)
    args=p.parse_args()
    root=args.first_stage_root.resolve()
    # Require both selection studies to have scored their frozen policies first.
    rep=read(root/'remaining_folds'/'results.json')
    read(root/'results.json')
    if digest(args.task1)!=read(root/'protocol.json')['archive_sha256']:
        raise ValueError('Unexpected training data')
    opened=[]
    with train_only(args.task1,opened):
        whole=load_training_task(args.task1)
    folds=[evaluate_fold((root if i<3 else root/'remaining_folds')/f'fold_{i}',whole) for i in range(5)]
    measured=[f for f in folds if f['available']]
    totals={}
    for group in ('all','lower_support','other'):
        totals[group]=dict(rows=sum(f[group]['rows'] for f in measured),
            correct={a:sum(f[group]['correct'][a] for f in measured) for a in ('original','semantic','permuted')},
            paired={k:{x:sum(f[group][k][x] for f in measured) for x in ('corrected','lost','changed')}
                    for k in ('semantic_vs_original','semantic_vs_permuted')})
    if len(measured)==5 and totals['all']['rows']!=rep['all_five']['rows']:
        raise ValueError('Unequal scoring population')
    result=dict(role='Exploratory after policy selection. Predictions from stored fifth-trial models and matching first-four leader on the identical, smaller inner fitting set; distinct from the full-refit policy results.',
                new_fits=0,new_encoder_forwards=0,opened=opened,folds=folds,totals=totals)
    write(root/'matched_backbone_diagnostic.json',result)
    print('Post-selection diagnostic',totals,flush=True)


if __name__=='__main__':
    main()
