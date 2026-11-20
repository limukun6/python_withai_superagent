"""Compare complete same-budget policies on grouped splits of released training data."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
from importlib.metadata import version

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from superagent.benchmark import run_one
from superagent.confirmation import shortlist
from superagent.data import DevelopmentTask, TrainingTask, load_training_task
from superagent.method_choice import MODES
from superagent.models import predict_with_model
from superagent.pretrained_choice import digest
from superagent.splits import effective_grouping, grouped_partition, overlap_audit, sample_groups, take
from tools.expanded_feedback_pilot import RESERVE_SEED, SEARCH_SEED
from tools.method_device_study import CAP, CAMPAIGN, OPTIONS
from tools.pretrained_study import read, train_only, write
from tools.tabular_encoder_study import paired


def expect(condition, message):
    if not condition:
        raise ValueError(message)


def receipt(folder):
    summary=read(folder/'summary.json')
    trials=read(folder/'trials.json')
    confirm=read(folder/'confirmation.json')
    selection=read(folder/'final_selection.json')
    first=shortlist(trials[:5],1)[0]['candidate']
    fits=summary['classifier_fit_accounting']['completed_classifier_fits']
    expect(len(trials)==7 and all(t['status']=='completed' for t in trials) and fits==14,'Incomplete fit budget')
    expect(summary['valid_policy_run'] and summary['refit_succeeded'] and summary['fallback_count']==0,'Invalid run')
    expect(not summary['test_accessed'] and not summary['provided_validation_available'],'Unexpected evaluation split')
    expect(summary['llm_calls']==0 and summary['usage_complete'] and summary['course_cost_rmb']<=CAP,'Usage check failed')
    expect(confirm['valid_confirmation'] and first in confirm['shortlist'],'Initial leader missing or confirmation failed')
    expect(digest(folder/'selected_model.joblib')==selection['model_sha256'],'Frozen model changed')
    return dict(model_sha256=selection['model_sha256'],winner=summary['selected_candidate'],fits=fits,
                extras=[t['candidate'] for t in trials[5:]],incumbent=first,seconds=summary['elapsed_seconds'],
                course_rmb=summary['course_cost_rmb'],api_rmb=summary['actual_cost_rmb'],calls=summary['llm_calls'],
                qwen_tokens=summary['tokens'],minilm_tokens=summary['downstream_feature_input_tokens'],
                decisions=[read(f) for f in sorted(folder.glob('method_decision_*.json'))])


def old_runs(roots, fingerprint, seed):
    locations={}
    for root in roots:
        root=root.resolve()
        protocol=read(root/'protocol.json')
        expect(protocol['archive_sha256']==fingerprint and protocol['seed']==seed and
               protocol['options']==OPTIONS and protocol['search_seed']==SEARCH_SEED and
               protocol['reserve_seed']==RESERVE_SEED,'Mismatched reference protocol')
        for i,split in enumerate(protocol['indices']):
            for arm in protocol['order'][i]:
                folder=(Path(protocol['locations'][i][arm]) if 'locations' in protocol else root/f'fold_{i}'/arm)
                if not any((folder/name).is_file() for name in ('frontier_receipt.json','receipt.json')):
                    continue
                key=(tuple(split['outer']),arm)
                if key in locations:expect(locations[key][0]==folder,'Conflicting reference arms')
                locations[key]=(folder,split['train'])
    return locations


def budget_ids(ledger,scope,roots,fresh):
    with sqlite3.connect(f'file:{ledger.resolve()}?mode=ro',uri=True) as db:
        rows=db.execute("SELECT DISTINCT json_extract(payload,'$.run_id') FROM requests WHERE scope=?",(scope,))
        earlier={name for (name,) in rows if name and any(Path(name).is_relative_to(p) for p in roots)}
    return sorted(earlier|{str(p) for p in fresh})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for field in ('archive','output','ledger','minilm-weights','weights','historical-protocol'):
        parser.add_argument('--'+field,type=Path,required=True)
    parser.add_argument('--arms',nargs='+',required=True,choices=('classic',*MODES))
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--folds',nargs='+',type=int,default=[1,2,3,4,5])
    parser.add_argument('--references',nargs='*',type=Path,default=[])
    parser.add_argument('--shared-budget',nargs='*',type=Path,default=[])
    parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    os.environ['SUPERAGENT_OFFLINE']='1'
    arms=args.arms
    expect(len(set(arms))==len(arms) and len(arms)>=2,'Supply at least two distinct arms in baseline-first order')
    expect(len(set(args.folds))==len(args.folds) and all(1<=f<=5 for f in args.folds),'Invalid fold numbers')
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=True)
    opened=[]
    with train_only(args.archive,opened):
        whole=load_training_task(args.archive)
    grouping=effective_grouping(whole.kind,'shared_text')
    keep,reserved,_=grouped_partition(whole,RESERVE_SEED,grouping)
    pool=TrainingTask(whole.kind,take(whole.X_train,keep),whole.y_train[keep],whole.spec)
    groups=sample_groups(pool,grouping)
    all_splits=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=args.seed).split(np.zeros(len(keep)),pool.y_train,groups))
    fingerprint=digest(args.archive)
    refs=old_runs(args.references,fingerprint,args.seed)
    splits=[]
    locations=[]
    orders=[]
    for fold in args.folds:
        train,outer=all_splits[fold-1]
        check=overlap_audit(pool,train,outer)
        expect(not check.get('search_duplicate_rows',0) and not check.get('search_rows_sharing_text',0),'Group leakage')
        row=dict(train=keep[train].tolist(),outer=keep[outer].tolist(),audit=check)
        splits.append(row)
        paths={}
        for arm in arms:
            ref=refs.get((tuple(row['outer']),arm))
            if ref:expect(ref[1]==row['train'],'Reference training rows differ')
            paths[arm]=str(ref[0] if ref else root/f'fold_{fold-1}'/arm)
        locations.append(paths)
        offset=(fold-1)%len(arms)
        orders.append(arms[offset:]+arms[:offset])
    repo=Path(__file__).resolve().parents[1]
    inputs=[*sorted((repo/'superagent').glob('*.py')),Path(__file__),repo/'tools/method_device_study.py',
            repo/'tools/expanded_feedback_pilot.py',repo/'tools/pretrained_study.py',repo/'tools/tabular_encoder_study.py']
    scope=next(t for t in read(args.historical_protocol)['tasks'] if t['package_sha256']==fingerprint)
    paid_roots=[p.resolve() for p in [root.parent,*args.shared_budget]]
    protocol=dict(archive_sha256=fingerprint,seed=args.seed,scope=scope['scope'],search_seed=SEARCH_SEED,
                  reserve_seed=RESERVE_SEED,reserved=len(reserved),fold_numbers=args.folds,indices=splits,opened=opened,
                  locations=locations,order=orders,options=OPTIONS,source={str(p.relative_to(repo)):digest(p) for p in inputs},
                  dependencies={k:version(k) for k in ('catboost','numpy','scikit-learn','torch','transformers')},
                  reference_protocols={str(p.resolve()):digest(p/'protocol.json') for p in args.references},
                  shared_budget_roots=list(map(str,paid_roots)),arm_cap=CAP,campaign_cap=CAMPAIGN,
                  promotion='Require at least 0.2 percentage points pooled gain and a strict majority of strict fold wins against every comparator. A candidate qualifying on explored data must replicate on a new grouped split before promotion.')
    if (root/'protocol.json').exists():expect(read(root/'protocol.json')==protocol,'Frozen protocol differs')
    else:write(root/'protocol.json',protocol)
    if args.freeze:
        print('Frozen',root,flush=True)
        return
    fresh=[Path(v) for block in locations for v in block.values() if Path(v).is_relative_to(root)]
    config=dict(ledger=str(args.ledger.resolve()),scope=scope['scope'],minilm_weights=str(args.minilm_weights.resolve()),
                tabular_weights=str(args.weights.resolve()),cap_rmb=CAP,campaign_cap_rmb=CAMPAIGN,
                previous_run_ids=budget_ids(args.ledger,scope['scope'],paid_roots,fresh),deferred_ids=scope['legacy_unknown_ids'])
    for i,split in enumerate(splits):
        task=DevelopmentTask(whole.kind,'training_partition',take(whole.X_train,split['train']),whole.y_train[split['train']],None,None,whole.spec)
        for arm in orders[i]:
            folder=Path(locations[i][arm])
            if folder.is_relative_to(root) and not (folder/'frontier_receipt.json').exists():
                expect(not folder.exists(),f'Incomplete run already exists at {folder}')
                print('Starting',args.folds[i],arm,flush=True)
                opts={} if arm=='classic' else dict(pretrained_config=config,method_choice=arm)
                run_one(task,'local_interleaved_routes',SEARCH_SEED,folder,None,**OPTIONS,**opts)
                result=receipt(folder)
                write(folder/'frontier_receipt.json',result)
                print('Frozen',args.folds[i],arm,result['extras'],result['winner'],flush=True)
            else:
                receipt(folder)
    totals={arm:0 for arm in arms}
    details=[]
    for i,split in enumerate(splits):
        held=split['outer']
        x=take(whole.X_train,held)
        truth=whole.y_train[held]
        pred={}
        evidence={}
        initial=None
        for arm in arms:
            folder=Path(locations[i][arm])
            rec=receipt(folder)
            first=[joblib.load(folder/f'trial_{k}.joblib')['search_predictions'] for k in range(1,6)]
            if initial is not None:
                for a,b in zip(initial,first,strict=True):np.testing.assert_array_equal(a,b)
            else:initial=first
            bundle=joblib.load(folder/'selected_model.joblib')
            expect(bundle['candidate']==rec['winner'],'Winner differs from saved bundle')
            pred[arm]=predict_with_model(whole.kind,bundle['model'],x)
            if not folder.is_relative_to(root):
                saved_predictions=folder.parent/'outer_predictions.json'
                if saved_predictions.exists():
                    historical=read(saved_predictions)
                    expect(historical['rows']==held,'Historical prediction indices differ')
                    np.testing.assert_array_equal(historical[arm],pred[arm])
                else:
                    prior=read(folder/'frontier_receipt.json')
                    expect(prior['model_sha256']==rec['model_sha256'],'Partial study model differs from its frozen receipt')
            correct=int(np.sum(pred[arm]==truth))
            totals[arm]+=correct
            evidence[arm]=dict(**rec,correct=correct,reused=not folder.is_relative_to(root))
        local=root/f'fold_{args.folds[i]-1}'
        local.mkdir(exist_ok=True)
        write(local/'outer_predictions.json',dict(rows=held,truth=truth.tolist(),**{k:p.tolist() for k,p in pred.items()}))
        comparisons={a:{b:paired(pred[b],pred[a],truth) for b in arms if b!=a} for a in arms}
        details.append(dict(fold=args.folds[i],rows=len(held),arms=evidence,comparisons=comparisons))
    rows=sum(f['rows'] for f in details)
    wins={a:{b:sum(f['arms'][a]['correct']>f['arms'][b]['correct'] for f in details) for b in arms if b!=a} for a in arms}
    qualify={a:all((totals[a]-totals[b])/rows>=.002 and w>len(details)/2 for b,w in matches.items()) for a,matches in wins.items()}
    result=dict(rows=rows,correct=totals,accuracy={a:v/rows for a,v in totals.items()},wins=wins,qualified=qualify,
                new_fits=sum(r['fits'] for f in details for r in f['arms'].values() if not r['reused']),
                new_course_rmb=sum(r['course_rmb'] for f in details for r in f['arms'].values() if not r['reused']),
                new_api_calls=sum(r['calls'] for f in details for r in f['arms'].values() if not r['reused']),
                folds=details)
    write(root/'results.json',result)
    print({k:v for k,v in result.items() if k!='folds'},flush=True)


if __name__=='__main__':main()
