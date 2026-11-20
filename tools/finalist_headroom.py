"""Measure selection headroom for a frozen finalist list and replay an honest vote rule."""
import argparse
import os
from pathlib import Path
import shutil

import joblib
import numpy as np

from superagent.benchmark import train_candidate
from superagent.confirmed_vote import decision, majority
from superagent.data import load_training_task
from superagent.models import predict_with_model
from superagent.minilm_features import FILES as MINILM_FILES, MiniLMTextClassifier, feature_usage
from superagent.pretrained_choice import digest
from superagent.splits import take
from tools.frontier_study import expect,receipt
from tools.pretrained_study import read,write,train_only
from tools.tabular_encoder_study import paired


def feature_weights(directory):
    if directory is None:return None
    directory=directory.resolve()
    names={name:digest(directory/name) for name in MINILM_FILES}
    expect(names=={name:item[1] for name,item in MINILM_FILES.items()},'MiniLM files changed')
    return names


def scoring_cache(destination,seed_caches):
    destination.mkdir(parents=True,exist_ok=True)
    target=destination/'minilm_category_vectors.sqlite'
    if not target.exists():
        for source in seed_caches:
            file=source/target.name
            if file.exists():
                shutil.copyfile(file,target)
                break
    return destination


def predict_finalist(kind,model,x,weights,cache):
    if isinstance(model,MiniLMTextClassifier):
        expect(weights is not None,'Supply verified MiniLM weights for a text finalist')
        model.bind_prediction_weights(weights,cache)
    return predict_with_model(kind,model,x)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ('archive','reference','output'):
        parser.add_argument('--'+arg,type=Path,required=True)
    parser.add_argument('--arm',required=True)
    parser.add_argument('--minilm-weights',type=Path)
    parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    os.environ['SUPERAGENT_OFFLINE']='1'
    os.environ.pop('SUPERAGENT_PRETRAINED_RUNTIME',None)
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=True)
    protocol=read(args.reference/'protocol.json')
    expect(protocol['archive_sha256']==digest(args.archive),'Archive hash differs')
    records=[]
    for idx,paths in enumerate(protocol['locations']):
        folder=Path(paths[args.arm])
        rec=receipt(folder)
        records.append(dict(folder=str(folder),model=rec['model_sha256'],history=digest(folder/'trials.json'),
                            confirmation=digest(folder/'confirmation.json'),payload=digest(folder/'refit_data.joblib')))
    repo=Path(__file__).resolve().parents[1]
    source=[*sorted((repo/'superagent').glob('*.py')),Path(__file__),repo/'tools/frontier_study.py']
    setup=dict(reference_sha256=digest(args.reference/'protocol.json'),arm=args.arm,records=records,
               source={str(p.relative_to(repo)):digest(p) for p in source},diagnostic_refits=2*len(records),
               policy='Compare the full-trained confirmation winner against a gated three-finalist vote using the two retained primary fits. Gate is frozen in confirmed_vote.py. Each alternative has 14 deployment fits.',
               ceiling='Also refit the other two finalists on all available fitting rows solely for a diagnostic ceiling. Report the best whole model per fold and the full-refit majority, labelled as diagnostics. Never select a deployable policy using outer labels.')
    if args.minilm_weights:setup['scoring_feature_weights']=feature_weights(args.minilm_weights)
    if (root/'protocol.json').exists():expect(read(root/'protocol.json')==setup,'Frozen diagnostic changed')
    else:write(root/'protocol.json',setup)
    if args.freeze:return
    opened=[]
    with train_only(args.archive,opened):whole=load_training_task(args.archive)
    for i,record in enumerate(records):
        folder=Path(record['folder'])
        dest=root/f'fold_{i}'
        dest.mkdir(exist_ok=True)
        history=read(folder/'trials.json')
        gate=decision(folder,history,read(folder/'confirmation.json'))
        lock=dest/'decision.json'
        if lock.exists():expect(read(lock)==gate,'Decision changed')
        else:write(lock,gate)
        # A cache miss cannot invoke any service. Fitting may consult only these already computed vectors.
        write(dest/'pretrained_runtime.json',{'minilm_weights':str(args.minilm_weights.resolve())} if args.minilm_weights else {})
        for path in [*folder.glob('method_*.json'),*folder.glob('*vectors.sqlite'),folder/'pretrained_decision.json',folder/'candidate_registry.json']:
            if path.is_file() and not (dest/path.name).exists():shutil.copyfile(path,dest/path.name)
        full=joblib.load(folder/'refit_data.joblib')
        indices=protocol['indices'][i]['train']
        np.testing.assert_array_equal(full['y_train'],whole.y_train[indices])
        saved=np.asarray(full['X_train'])
        expected=np.asarray(take(whole.X_train,indices))
        expect(saved.shape==expected.shape and saved.dtype==expected.dtype,'Training feature structure differs')
        for a,b in zip(saved.flat,expected.flat,strict=True):
            if type(a) is not type(b) or repr(a)!=repr(b):
                raise ValueError('Training row order or feature value differs')
        full.pop('feature_cache_dir',None)
        payload=dest/'diagnostic_training.joblib'
        if not payload.exists():joblib.dump(full,payload)
        hashes={}
        for num,name in enumerate(gate['components'][1:],1):
            target=dest/f'full_{num}.joblib'
            if not target.exists():
                model,metric=train_candidate(payload,name,target,480)
                expect(model is not None and metric['classifier_fits']==1,str(metric))
            hashes[name]=digest(target)
        expect(not (dest/'llm_usage.jsonl').exists() and not (dest/'downstream_usage.jsonl').exists(),'Unexpected encoder forward')
        write(dest/'fits.json',hashes)
        print('Frozen finalist models',i+1,gate['adopt'],flush=True)
    results=[]
    for i,record in enumerate(records):
        folder=Path(record['folder']); dest=root/f'fold_{i}'
        keys={r['candidate']:r['trial'] for r in read(folder/'trials.json')}
        gate=read(dest/'decision.json')
        names=gate['components']
        held=protocol['indices'][i]['outer']
        x=take(whole.X_train,held); truth=whole.y_train[held]
        cache=scoring_cache(dest/'diagnostic_prediction_features',[folder.parent/(args.arm+'_prediction_features')]) if args.minilm_weights else None
        model=joblib.load(folder/'selected_model.joblib')['model']
        anchor=predict_finalist(whole.kind,model,x,args.minilm_weights,cache)
        np.testing.assert_array_equal(anchor,read(folder.parent/'outer_predictions.json')[args.arm])
        search_predictions=[anchor]
        full_predictions=[anchor]
        for num,name in enumerate(names[1:],1):
            search=joblib.load(folder/f'trial_{keys[name]}.joblib')['model']
            search_predictions.append(predict_finalist(whole.kind,search,x,args.minilm_weights,cache))
            target=dest/f'full_{num}.joblib'
            expect(digest(target)==read(dest/'fits.json')[name],'Refitted finalist changed')
            fitted=joblib.load(target)['model']
            full_predictions.append(predict_finalist(whole.kind,fitted,x,args.minilm_weights,cache))
        vote=majority(search_predictions)
        actual=vote if gate['adopt'] else anchor
        best=max(int(np.sum(p==truth)) for p in full_predictions)
        vectors=dict(incumbent=anchor,gated=actual,always_vote=vote,full_vote=majority(full_predictions))
        write(dest/'outer_predictions.json',dict(rows=held,truth=truth.tolist(),**{k:v.tolist() for k,v in vectors.items()},
                                                 full_finalists={k:v.tolist() for k,v in zip(names,full_predictions,strict=True)}))
        results.append(dict(fold=protocol['fold_numbers'][i],rows=len(held),adopted=gate['adopt'],
                       correct={k:int(np.sum(v==truth)) for k,v in vectors.items()},oracle_single=best,
                       gated_vs_incumbent=paired(anchor,actual,truth),finalists=names))
    totals={k:sum(f['correct'][k] for f in results) for k in results[0]['correct']}
    usages=[feature_usage(root/f'fold_{i}'/'diagnostic_prediction_features') for i in range(len(results))]
    summary=dict(folds=results,correct=totals,oracle_single=sum(f['oracle_single'] for f in results),
                 rows=sum(f['rows'] for f in results),training_members=opened,additional_diagnostic_fits=2*len(results),
                 api_calls=0,additional_encoder_forwards=sum(u['batches'] for u in usages),
                 additional_minilm_input_tokens=sum(u['input_tokens'] for u in usages))
    write(root/'results.json',summary)
    print({k:v for k,v in summary.items() if k!='folds'},flush=True)


if __name__=='__main__':main()
