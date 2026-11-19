"""Training-only, paired comparison of classic search, fixed expansion and feedback allocation."""
import argparse
import json
from pathlib import Path
import platform
from importlib.metadata import version
import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from superagent.data import DevelopmentTask,TrainingTask,load_training_task
from superagent.splits import grouped_partition,sample_groups,take,overlap_audit
from superagent.benchmark import run_one
from superagent.confirmation import shortlist
from superagent.models import predict_with_model
from superagent.pretrained_choice import digest
from superagent.minilm_features import MODEL as MINI_MODEL, FILES as MINI_FILES
from superagent.tabular_encoder import FILES as QWEN_FILES
from tools.expanded_feedback_pilot import RUN_OPTIONS,RESERVE_SEED,SEARCH_SEED
from tools.pretrained_study import train_only,write,read
from tools.tabular_encoder_study import paired,usage

ARMS=('classic','fixed','adaptive')
OPTIONS=RUN_OPTIONS | dict(max_trials=7,confirmation_top_k=3,trial_timeout=480,time_limit=1600)
SEED=46877
CAP=.012
CAMPAIGN=.06


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('archive','output','ledger','weights','minilm-weights','historical-protocol'):
        p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args()
    root=a.output.resolve()
    root.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[1]
    opened=[]
    with train_only(a.archive,opened):whole=load_training_task(a.archive)
    keep,reserve,_=grouped_partition(whole,RESERVE_SEED)
    pool=TrainingTask(whole.kind,take(whole.X_train,keep),whole.y_train[keep],whole.spec)
    groups=sample_groups(pool)
    splits=list(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED).split(np.zeros(len(keep)),pool.y_train,groups))
    ids=[str(root/f'fold_{i}'/arm) for i in range(len(splits)) for arm in ARMS]
    archive_sha=digest(a.archive)
    scope=next(r for r in read(a.historical_protocol)['tasks'] if r['package_sha256']==archive_sha and r['scope']=='task1')
    sources=[*sorted((repo/'superagent').glob('*.py')),Path(__file__),repo/'tools/tabular_encoder_study.py',repo/'tools/pretrained_study.py',repo/'tools/expanded_feedback_pilot.py']
    orders=[ARMS[i%3:]+ARMS[:i%3] for i in range(5)]
    plan=dict(seed=SEED,search_seed=SEARCH_SEED,reserve_seed=RESERVE_SEED,reserve_count=len(reserve),options=OPTIONS,
        order=orders,opened=opened,archive_sha256=archive_sha,source={str(p.relative_to(repo)):digest(p) for p in sources},
        dependencies={k:version(k) for k in ('torch','numpy','scikit-learn','catboost','transformers')},platform=platform.platform(),
        arm_cap=CAP,campaign_cap=CAMPAIGN,minilm_files=MINI_FILES,qwen_files=QWEN_FILES,
        fixed_policy="Try frozen MiniLM followed by Qwen for compatible categorical tables. Both attach to the five-trial leader. Adaptive may instead train a neural network, learn co-occurrence, or continue ordinary search.",
        secondary="Compare both frozen encoder trials and the matched ordinary incumbent on identical fitting rows. Replay decisions with outcome evidence removed. Audit actual CPU vs accelerator training on the first inner split after all winners freeze.",
        purpose='Compare complete policies. Adoption requires adaptive beating both controls on pooled rows and on at least three of five folds against each; otherwise keep existing default. All methods chosen without outer labels.',
        indices=[dict(train=keep[t].tolist(),outer=keep[h].tolist(),audit=overlap_audit(pool,t,h)) for t,h in splits])
    assert len(set(np.concatenate([h for _,h in splits])))==len(keep)
    assert all(f['audit']['search_duplicate_rows']==0 for f in plan['indices'])
    if (root/'protocol.json').exists():
        if read(root/'protocol.json')!=json.loads(json.dumps(plan)):raise ValueError('Frozen protocol changed')
    else:write(root/'protocol.json',plan)
    for i,(train,held) in enumerate(splits):
        task=DevelopmentTask(pool.kind,'training_partition',take(pool.X_train,train),pool.y_train[train],None,None,pool.spec)
        for arm in orders[i]:
            dest=root/f'fold_{i}'/arm
            if (dest/'receipt.json').exists():
                if digest(dest/'selected_model.joblib')!=read(dest/'receipt.json')['sha256']:raise ValueError('Changed model')
                continue
            print('Starting fold',i+1,arm,flush=True)
            config=dict(ledger=str(a.ledger.resolve()),scope='task1',tabular_weights=str(a.weights.resolve()),minilm_weights=str(a.minilm_weights.resolve()),
                        cap_rmb=CAP,campaign_cap_rmb=CAMPAIGN,previous_run_ids=ids,deferred_ids=scope['legacy_unknown_ids'])
            opts={} if arm=='classic' else dict(method_choice=arm,pretrained_config=config)
            summary=run_one(task,'local_interleaved_routes',SEARCH_SEED,dest,None,**OPTIONS,**opts)
            trials=read(dest/'trials.json')
            confirmation=read(dest/'confirmation.json')
            incumbent=shortlist(trials[:5],1)[0]['candidate']
            fits=summary['classifier_fit_accounting']['completed_classifier_fits']
            receipt=dict(winner=summary['selected_candidate'],sha256=digest(dest/'selected_model.joblib'),fits=fits,
                trials=[dict(id=r['candidate'],score=r.get('search_score'),status=r['status'],error=r.get('error'),
                             train_metadata=r.get('training_metadata')) for r in trials],
                finalists=confirmation['shortlist'],means={r['candidate']:r['mean_score'] for r in confirmation['candidates']},
                incumbent=incumbent,seconds=summary['elapsed_seconds'],usage=usage(dest,a.ledger),
                decisions=[read(f) for f in sorted(dest.glob('method_decision_*.json'))],
                complete=(len(trials)==7 and fits==14 and summary['refit_succeeded'] and summary['valid_policy_run']
                    and not summary['provided_validation_available'] and not summary['test_accessed'] and summary['usage_complete']
                    and summary['fallback_count']==0 and incumbent in confirmation['shortlist'] and confirmation['valid_confirmation']))
            features_file=dest/'downstream_usage.jsonl'
            features=[json.loads(line) for line in features_file.read_text().splitlines()] if features_file.exists() else []
            assert all(r['model']==MINI_MODEL and r['api_rmb']==0. and not r['remote_calls'] for r in features)
            assert summary['llm_calls']==0 and receipt['usage']['course_rmb']<=CAP
            receipt['downstream_minilm']=dict(forwards=len(features),input_tokens=sum(r['input_tokens'] for r in features),
                                               sequences=sum(r['sequences'] for r in features),api_rmb=0.)
            if not receipt['complete']:
                write(dest/'incomplete.json',receipt)
                raise ValueError(f'Incomplete arm {dest}; outer results stay sealed')
            write(dest/'receipt.json',receipt)
            print('Frozen',i+1,arm,[r['id'] for r in receipt['trials'][5:]],receipt['winner'],flush=True)
    results=[]
    for i,(train,held) in enumerate(splits):
        truth=pool.y_train[held]
        dest=root/f'fold_{i}'
        predictions={}
        arms={}
        for arm in ARMS:
            folder=dest/arm
            rec=read(folder/'receipt.json')
            assert digest(folder/'selected_model.joblib')==rec['sha256']==read(folder/'final_selection.json')['model_sha256']
            model=joblib.load(folder/'selected_model.joblib')
            pred=predict_with_model(pool.kind,model['model'],take(pool.X_train,held))
            assert pred.shape==truth.shape
            predictions[arm]=pred
            arms[arm]=dict(**rec,correct=int((pred==truth).sum()),accuracy=float((pred==truth).mean()))
            if rec['trials'][:5]!=arms['classic']['trials'][:5]:raise ValueError('Different initial evidence')
        gains={a:paired(predictions[a],predictions['adaptive'],truth) for a in ('classic','fixed')}
        results.append(dict(rows=len(truth),arms=arms,adaptive_vs=gains))
        write(dest/'outer_predictions.json',dict(rows=keep[held].tolist(),truth=truth.tolist(),**{k:v.tolist() for k,v in predictions.items()}))
    totals={a:sum(f['arms'][a]['correct'] for f in results) for a in ARMS}
    wins={a:sum(f['arms']['adaptive']['correct']>f['arms'][a]['correct'] for f in results) for a in ('classic','fixed')}
    cost=sum(v['usage']['course_rmb'] for f in results for v in f['arms'].values())
    assert cost<=CAMPAIGN
    out=dict(rows=len(keep),correct=totals,accuracy={a:c/len(keep) for a,c in totals.items()},folds=results,
             wins=wins,promote=all(totals['adaptive']>totals[a] and wins[a]>=3 for a in wins),course_rmb=cost)
    write(root/'results.json',out)
    print({k:v for k,v in out.items() if k!='folds'},flush=True)


if __name__=='__main__':main()
