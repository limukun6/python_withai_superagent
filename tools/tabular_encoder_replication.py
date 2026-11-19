"""Complete the two remaining folds of the frozen category transfer experiment."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from superagent.data import DevelopmentTask, TrainingTask, load_training_task
from superagent.models import predict_with_model
from superagent.pretrained_choice import digest
from superagent.splits import grouped_partition, overlap_audit, sample_groups, take
from .expanded_feedback_pilot import RESERVE_SEED
from .pretrained_study import read, write, train_only
from .tabular_encoder_study import ARM_CAP, CAMPAIGN_CAP, ORDER, OUTER_SEED, freeze_run, usage, paired

REMAINING = {3:('original','semantic','permuted'), 4:('permuted','original','semantic')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('task1','first-stage-root','ledger','weights','historical-protocol'):
        p.add_argument('--'+field,type=Path,required=True)
    args = p.parse_args()
    first = args.first_stage_root.resolve()
    root = first/'remaining_folds'
    root.mkdir(exist_ok=True)
    protocol = read(first/'protocol.json')
    primary = read(first/'results.json')
    repo = Path(__file__).resolve().parents[1]
    if any(digest(repo/name)!=sha for name, sha in protocol['source'].items()):
        raise ValueError('The first-stage decision or training code changed')
    if digest(args.task1)!=protocol['archive_sha256'] or digest(args.historical_protocol)!=protocol['historical_deferral_sha256']:
        raise ValueError('Archive or ledger scope differs from the first stage')
    historical = next(r for r in read(args.historical_protocol)['tasks'] if r['scope']=='task1' and r['package_sha256']==protocol['archive_sha256'])
    opened = []
    with train_only(args.task1, opened):
        whole = load_training_task(args.task1)
    keep, reserve, _ = grouped_partition(whole, RESERVE_SEED)
    dev = TrainingTask(whole.kind, take(whole.X_train, keep), whole.y_train[keep], whole.spec)
    groups = sample_groups(dev)
    data, splits, all_held = {}, {}, []
    for i, (train, held) in enumerate(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=OUTER_SEED).split(np.zeros(len(keep)),dev.y_train,groups)):
        record = dict(train=keep[train].tolist(), outer=keep[held].tolist(), audit=overlap_audit(dev,train,held))
        all_held.extend(record['outer'])
        if i < 3 and record != protocol['splits'][i]:
            raise ValueError('Previously declared fold identities changed')
        if record['audit']['search_duplicate_rows'] or set(dev.y_train[train])!=set(dev.y_train[held]):
            raise ValueError('Invalid grouped evaluation')
        if i in REMAINING:
            task = DevelopmentTask(dev.kind,'training_partition',take(dev.X_train,train),dev.y_train[train],None,None,dev.spec)
            data[i] = (task,take(dev.X_train,held),dev.y_train[held])
            splits[str(i)] = record
    if len(set(all_held))!=len(all_held) or set(all_held)!=set(keep) or set(all_held).intersection(reserve):
        raise ValueError('The five outer folds must exhaust only the non-reserved pool')
    plan = dict(first_protocol_sha256=digest(first/'protocol.json'), first_results_sha256=digest(first/'results.json'),
        replication_code_sha256=digest(Path(__file__)), order=REMAINING, splits=splits, opened=opened,
        options=protocol['options'], cap_rmb=ARM_CAP, shared_campaign_cap_rmb=CAMPAIGN_CAP,
        outcomes='After all six additional winners freeze, report both folds, and the full five-fold result. Look for pooled semantic accuracy above both original and permuted in these two remaining folds. All first-stage model rules and hyperparameters are frozen.')
    if (root/'protocol.json').exists():
        if read(root/'protocol.json')!=json.loads(json.dumps(plan)):
            raise ValueError('Replication protocol mismatch')
    else:
        write(root/'protocol.json',plan)
    run_ids = [str(first/f'fold_{i}'/a) for i, arms in enumerate(ORDER) for a in arms]
    run_ids += [str(root/f'fold_{i}'/a) for i, arms in REMAINING.items() for a in arms]
    for i, arms in REMAINING.items():
        for arm in arms:
            dest = root/f'fold_{i}'/arm
            if (dest/'frozen.json').exists():
                receipt = read(dest/'frozen.json')
                if digest(dest/'selected_model.joblib')!=receipt['model_sha256']:
                    raise ValueError('Frozen model changed')
            else:
                config = dict(tabular_weights=str(args.weights.resolve()), ledger=str(args.ledger.resolve()),scope='task1',
                              cap_rmb=ARM_CAP,campaign_cap_rmb=CAMPAIGN_CAP,previous_run_ids=run_ids,
                              deferred_ids=historical['legacy_unknown_ids'],level_assignment=arm)
                print('Starting remaining fold',i,arm,flush=True)
                receipt = freeze_run(data[i][0],dest,config if arm!='original' else None)
            if not receipt['complete']:
                raise ValueError(f'Incomplete replication. Inspect {dest}')
            print('Frozen',i,arm,receipt['winner'],flush=True)
    results = []
    for i, arms in REMAINING.items():
        _, features, labels = data[i]
        records, preds = {}, {}
        for arm in arms:
            dest = root/f'fold_{i}'/arm
            f = read(dest/'frozen.json')
            model_path = dest/'selected_model.joblib'
            if digest(model_path)!=f['model_sha256']:
                raise ValueError('Frozen model changed')
            bundle = joblib.load(model_path)
            values = predict_with_model(bundle['kind'],bundle['model'],features)
            if values.shape!=labels.shape or bundle['candidate']!=f['winner']:
                raise ValueError('Invalid frozen winner')
            preds[arm] = values
            if records and f['trials'][:4]!=next(iter(records.values()))['trials'][:4]:
                raise ValueError('First-four evidence differs between arms')
            correct = int(np.sum(values==labels))
            records[arm] = dict(**f, correct=correct, accuracy=correct/len(labels), usage=usage(dest,args.ledger),
                                decision=read(dest/'pretrained_decision.json') if arm!='original' else None)
        results.append(dict(rows=len(labels),arms=records, semantic_vs_original=paired(preds['original'],preds['semantic'],labels),
             permuted_vs_original=paired(preds['original'],preds['permuted'],labels),
             semantic_vs_permuted=paired(preds['permuted'],preds['semantic'],labels)))
        write(root/f'fold_{i}'/'outer_predictions.json',dict(rows=splits[str(i)]['outer'],truth=labels.tolist(),
              **{a:y.tolist() for a,y in preds.items()}))
    additional = {arm:sum(f['arms'][arm]['correct'] for f in results) for arm in ORDER[0]}
    totals = {arm:additional[arm]+primary['correct'][arm] for arm in additional}
    extra_n = sum(f['rows'] for f in results)
    n = primary['rows']+extra_n
    course = sum(r['usage']['course_rmb'] for f in results for r in f['arms'].values())
    if course+primary['course_rmb'] > CAMPAIGN_CAP:
        raise ValueError('Combined allowance exceeded')
    write(root/'results.json',dict(folds=results,rows=extra_n,correct=additional,
          replication_positive=all(additional['semantic']>additional[a] for a in ('original','permuted')),
          course_rmb=course, all_five=dict(rows=n, correct=totals, accuracy={a:c/n for a,c in totals.items()},
          course_rmb=course+primary['course_rmb'])))
    print('Complete remaining folds',additional,'full five',totals,flush=True)


if __name__=='__main__':
    main()
