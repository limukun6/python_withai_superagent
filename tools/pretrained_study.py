"""Single frozen, paired, train-only study on a common outer split."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import platform
import sqlite3
import time
import zipfile

import joblib
import numpy as np
from .expanded_feedback_pilot import RUN_OPTIONS, SEARCH_SEED, sha
from .search_budget_study import outer_partition
from superagent.benchmark import run_one
from superagent.models import predict_with_model
from superagent.pretrained_choice import MODEL, FILES
from superagent.pricing import resolve_pricing

SEED = 44231
OPTIONS = {**RUN_OPTIONS, 'time_limit': 2000, 'trial_timeout': 1500, 'max_trials': 5, 'confirmation_top_k': 2}
CAP = .075
ORDER = [('task2','choice'),('task2','control'),('task1','control'),('task1','choice')]


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def read(path):
    return json.loads(path.read_text())


@contextmanager
def train_only(path, opened):
    original = zipfile.ZipFile.open
    def checked(archive, name, *args, **kwargs):
        if Path(archive.filename).resolve() == path.resolve():
            member = name if isinstance(name,str) else name.filename
            if Path(member).name not in {'C_train.npy','y_train.npy','train.csv'}:
                raise RuntimeError(f'Non-training member refused: {member}')
            opened.append(member)
        return original(archive, name, *args, **kwargs)
    zipfile.ZipFile.open = checked
    try:
        yield
    finally:
        zipfile.ZipFile.open = original


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for item in ('task1','task2','output','ledger','weights'):
        p.add_argument('--'+item, required=True, type=Path)
    p.add_argument('--prior-paired-root', type=Path,help='Frozen earlier pilot, to enforce the shared campaign limit in the same task ledger')
    p.add_argument('--historical-protocol',type=Path,help='Existing user-deferred requests; no new unknown is permitted')
    args=p.parse_args()
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=True)
    repo=Path(__file__).resolve().parents[1]
    historical=read(args.historical_protocol)['tasks'] if args.historical_protocol else []
    data={}
    partitions={}
    accesses={}
    for name in ('task1','task2'):
        path=getattr(args,name)
        accesses[name]=[]
        with train_only(path,accesses[name]):
            task, features, labels, partitions[name]=outer_partition(path, SEED)
        data[name]=(task,features,labels)
    protocol=dict(seed=SEED, search_seed=SEARCH_SEED, options=OPTIONS, cap_per_arm_rmb=CAP,
        campaign_cap_per_task_rmb=.15,
        action='Retain all four original actions. At slot five, optionally use one pretrained candidate when >=2 successful originals share >=max(8,5%) errors on text. Top two of five always include the original four-step leader. Confirm both across three splits; refit winner.',
        order=ORDER, partitions=partitions, opened=accesses, weights=FILES, price=resolve_pricing(MODEL).metadata(),
        source={str(k.relative_to(repo)):sha(k) for k in [*sorted((repo/'superagent').glob('*.py')),Path(__file__),repo/'tools/search_budget_study.py',repo/'tools/expanded_feedback_pilot.py']},
        archives={k:sha(getattr(args,k)) for k in data}, python=platform.python_version(),
        previous_root=str(args.prior_paired_root.resolve()) if args.prior_paired_root else None, historical_deferral_protocol=sha(args.historical_protocol) if args.historical_protocol else None,
        evaluation='First paired train-only outer fold; no promotion from one seed.')
    protocol_path=root/'protocol.json'
    if protocol_path.exists():
        if read(protocol_path)!=json.loads(json.dumps(protocol)):
            raise ValueError('Frozen protocol mismatch; use a new output directory')
    else:
        write(protocol_path,protocol)
    for name,arm in ORDER:
        dest=root/name/arm
        if (dest/'frozen.json').exists():
            if sha(dest/'selected_model.joblib')!=read(dest/'frozen.json')['model_sha256']:
                raise ValueError('Invalid model on resume')
            continue
        task,_,_=data[name]
        deferred=[v for v in historical if v['scope']==name and v['package_sha256']==protocol['archives'][name]]
        config=dict(weights=str(args.weights.resolve()),ledger=str(args.ledger.resolve()),scope=name,cap_rmb=CAP,
                    deferred_ids=deferred[0]['legacy_unknown_ids'] if deferred else [],
                    campaign_cap_rmb=.15, previous_run_ids=[str(args.prior_paired_root.resolve()/name/a) for a in ('choice','control')] if args.prior_paired_root else [])
        print('Starting',name,arm,flush=True)
        summary=run_one(task,'local_interleaved_routes',SEARCH_SEED,dest,None,**OPTIONS,
                        pretrained_config=config if arm=='choice' else None)
        trials=read(dest/'trials.json')
        frozen=dict(model_sha256=sha(dest/'selected_model.joblib'), candidate=summary['selected_candidate'],
            fits=summary['classifier_fit_accounting']['completed_classifier_fits'], elapsed=summary['elapsed_seconds'],
            trials=[(r['candidate'],r['status'],r.get('search_score')) for r in trials],
            failure_details=[r.get('error') for r in trials if r['status']!='completed'],
            complete=summary['trials_attempted']==5 and summary['failed_trials']==0 and summary['refit_succeeded']
                and summary['confirmation']['valid_confirmation'] and summary['fallback_count']==0)
        if frozen['model_sha256']!=read(dest/'final_selection.json')['model_sha256']:
            raise ValueError('Final model hash mismatch')
        write(dest/'frozen.json',frozen)
        print(name,arm,'frozen',frozen['candidate'],'complete',frozen['complete'],flush=True)
    report={}
    for name,(_,features,labels) in data.items():
        arms={}
        predictions={}
        for arm in ('control','choice'):
            dest=root/name/arm
            frozen=read(dest/'frozen.json')
            bundle=joblib.load(dest/'selected_model.joblib')
            predictions[arm]=predict_with_model(bundle['kind'],bundle['model'],features)
            if predictions[arm].shape!=labels.shape:
                raise ValueError('Prediction dimensions differ')
            score=int(np.sum(predictions[arm]==labels))
            logs=[json.loads(s) for s in (dest/'llm_usage.jsonl').read_text().splitlines()]
            local=[r for r in logs if r['event']=='local_forward']
            tokens=sum(r['input_tokens'] for r in local)
            with sqlite3.connect(args.ledger) as db:
                rows=db.execute("SELECT id,status,course_units FROM requests WHERE json_extract(payload,'$.run_id')=?",(str(dest),)).fetchall()
            cost=resolve_pricing(MODEL).course_cost(tokens,0)
            if {r[0] for r in rows}!={r['request_id'] for r in local} or any(r[1]!='known' for r in rows) or abs(sum(r[2] for r in rows)/1e9-cost)>1e-10 or cost>CAP:
                raise ValueError('Local inference ledger mismatch')
            arms[arm]=dict(**frozen,correct=score, accuracy=score/len(labels),forwards=len(local),input_tokens=tokens,
                output_tokens=0,course_rmb=cost, remote_model_calls=0,decision=read(dest/'pretrained_decision.json') if arm=='choice' else None)
        for arm_name, result in arms.items():
            original = sorted(result['trials'][:4], key=lambda row:(-row[2],row[0]))[0][0]
            finalists = read(root/name/arm_name/'confirmation.json')['shortlist']
            if original not in finalists or len(finalists)!=2 or result['fits']!=10:
                raise ValueError('Original incumbent or equal fit budget was lost')
            result.update(protected_incumbent=original, finalists=finalists)
        if arms['control']['trials'][:4]!=arms['choice']['trials'][:4]:
            raise ValueError('Shared observations diverged before the decision')
        first,second=predictions['control'],predictions['choice']
        report[name]=dict(rows=len(labels),arms=arms,corrected=int(np.sum((first!=labels)&(second==labels))),
            lost=int(np.sum((first==labels)&(second!=labels))),changed=int(np.sum(first!=second)))
        write(root/name/'outer_predictions.json',dict(rows=partitions[name]['held_rows'],truth=labels.tolist(),
            control=first.tolist(),choice=second.tolist()))
    write(root/'results.json',report)
    print('Completed',root/'results.json',flush=True)


if __name__=='__main__':
    main()
