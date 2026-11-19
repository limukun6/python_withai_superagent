"""Three predeclared disjoint outer folds of the released training partition only."""
import argparse
import json
from pathlib import Path
import sqlite3

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from superagent.benchmark import run_one
from superagent.data import DevelopmentTask, TrainingTask, load_training_task
from superagent.models import predict_with_model
from superagent.pretrained_choice import digest
from superagent.pricing import resolve_pricing
from superagent.splits import grouped_partition, overlap_audit, sample_groups, take
from superagent.tabular_encoder import MODEL, FILES
from .expanded_feedback_pilot import RUN_OPTIONS, RESERVE_SEED, SEARCH_SEED
from .pretrained_study import train_only, read, write

OUTER_SEED = 45317
ORDER = (('semantic', 'original', 'permuted'), ('original', 'permuted', 'semantic'), ('permuted', 'semantic', 'original'))
OPTIONS = {**RUN_OPTIONS, 'max_trials': 5, 'confirmation_top_k': 2, 'time_limit': 2400, 'trial_timeout': 1800}
ARM_CAP, CAMPAIGN_CAP = .02, .075


def partitions(path):
    opened = []
    with train_only(path, opened):
        whole = load_training_task(path)
    if whole.kind != 'categorical_tabular':
        raise ValueError('This comparative study requires a categorical table')
    keep, reserve, _ = grouped_partition(whole, RESERVE_SEED)
    dev = TrainingTask(whole.kind, take(whole.X_train, keep), whole.y_train[keep], whole.spec)
    groups = sample_groups(dev)
    splits = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=OUTER_SEED)
    data, indices = [], []
    for i, (train, held) in enumerate(splits.split(np.zeros(len(keep)), dev.y_train, groups)):
        if i >= len(ORDER):
            break
        audit = overlap_audit(dev, train, held)
        if audit['search_duplicate_rows'] or set(dev.y_train[train]) != set(dev.y_train[held]):
            raise ValueError('Invalid outer split')
        task = DevelopmentTask(dev.kind, 'training_partition', take(dev.X_train, train), dev.y_train[train], None, None, dev.spec)
        data.append((task, take(dev.X_train, held), dev.y_train[held]))
        indices.append(dict(train=keep[train].tolist(), outer=keep[held].tolist(), audit=audit))
    flat = [i for fold in indices for i in fold['outer']]
    if len(set(flat)) != len(flat) or set(flat).intersection(reserve):
        raise ValueError('Outer rows overlap another fold or the reserved data')
    return data, indices, opened, len(reserve)


def freeze_run(task, dest, config):
    summary = run_one(task, 'local_interleaved_routes', SEARCH_SEED, dest, None, **OPTIONS, pretrained_config=config)
    trials = read(dest/'trials.json')
    confirmation = read(dest/'confirmation.json')
    receipt = dict(winner=summary['selected_candidate'], fits=summary['classifier_fit_accounting']['completed_classifier_fits'],
                   elapsed_seconds=summary['elapsed_seconds'], finalists=confirmation['shortlist'],
                   model_sha256=digest(dest/'selected_model.joblib'), failures=summary['failed_trials'],
                   trials=[dict(candidate=r['candidate'], score=r.get('search_score'), status=r['status'],
                                error=r.get('error')) for r in trials],
                   confirmation_means={r['candidate']:r['mean_score'] for r in confirmation['candidates']})
    if receipt['model_sha256'] != read(dest/'final_selection.json')['model_sha256']:
        raise ValueError('Selected artifact hash differs from final selection')
    original = min((r for r in trials[:4] if r['status']=='completed'), key=lambda r:(-r['search_score'],r['candidate']))['candidate']
    receipt['protected_incumbent'] = original
    receipt['complete'] = (len(trials)==5 and summary['failed_trials']==0 and summary['refit_succeeded']
                           and summary['fallback_count']==0 and summary['valid_policy_run'] and not summary['test_accessed']
                           and confirmation['valid_confirmation'] and len(receipt['finalists'])==2
                           and original in receipt['finalists'] and receipt['fits']==10)
    write(dest/'frozen.json', receipt)
    return receipt


def usage(dest, ledger):
    logs = [json.loads(s) for s in (dest/'llm_usage.jsonl').read_text().splitlines()]
    forwards = [r for r in logs if r['event']=='local_forward']
    if any(r.get('model') != MODEL or r['output_tokens'] or not r['usage_known'] for r in forwards):
        raise ValueError('Unexpected model usage')
    with sqlite3.connect(ledger) as db:
        rows = db.execute("SELECT id,status,course_units,payload FROM requests WHERE scope=? AND json_extract(payload,'$.run_id')=?",
                          ('task1', str(dest))).fetchall()
    tokens = sum(r['input_tokens'] for r in forwards)
    cost = resolve_pricing(MODEL).course_cost(tokens, 0)
    if ({r[0] for r in rows} != {r['request_id'] for r in forwards} or any(r[1]!='known' for r in rows)
            or abs(sum(r[2] for r in rows)/1e9-cost)>1e-10 or cost>ARM_CAP):
        raise ValueError('Local forwards and ledger disagree')
    return dict(forwards=len(forwards), sequences=sum(r['sequences'] for r in forwards),
                input_tokens=tokens, output_tokens=0, course_rmb=cost, actual_api_rmb=0., remote_inference_calls=0)


def paired(original, other, truth):
    return dict(corrected=int(np.sum((original!=truth)&(other==truth))), lost=int(np.sum((original==truth)&(other!=truth))),
                changed=int(np.sum(original!=other)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ('task1','output','ledger','weights','historical-protocol'):
        p.add_argument('--'+field, type=Path, required=True)
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    data, indices, opened, reserved_count = partitions(args.task1)
    archive_sha = digest(args.task1)
    historical = [t for t in read(args.historical_protocol)['tasks'] if t['scope']=='task1' and t['package_sha256']==archive_sha]
    if len(historical) != 1:
        raise ValueError('Historical budget scope does not match the released archive')
    source = [*sorted((repo/'superagent').glob('*.py')), Path(__file__), repo/'tools/expanded_feedback_pilot.py', repo/'tools/pretrained_study.py']
    protocol = dict(outer_seed=OUTER_SEED, outer_folds=[0,1,2], reserve_seed=RESERVE_SEED, reserved_rows=reserved_count,
        search_seed=SEARCH_SEED, splits=indices, opened=opened, archive_sha256=archive_sha,
        order=ORDER, options=OPTIONS, arm_cap_rmb=ARM_CAP, campaign_cap_rmb=CAMPAIGN_CAP,
        price=resolve_pricing(MODEL).metadata(), weights=FILES, source={str(s.relative_to(repo)):digest(s) for s in source},
        historical_deferral_sha256=digest(args.historical_protocol),
        decision='4 original trials; at slot 5 the optional arm can attach named-level features to the observed leader. Gate: >=2 successful originals, shared errors >=max(8,ceil(5%)), >=max(4,ceil(10% of shared errors)) with a lower-support named level. Confirm top two, refit winner. Original arm retains ordinary fifth proposal.',
        negative_control='Same encoder, folds, backbone, PCA, dimensions, and features; cyclically permute the coordinates assigned to distinct category labels separately per column. No fixed points.',
        outcome='Score the selected refitted policy once on disjoint outer rows after all 9 winners freeze. Promotion threshold: semantic improves over original on at least 2/3 folds with pooled gain >=0.2 percentage points, no fold loses over 1 point, and pooled accuracy exceeds permuted.')
    protocol_file = root/'protocol.json'
    if protocol_file.exists():
        if read(protocol_file) != json.loads(json.dumps(protocol)):
            raise ValueError('Frozen protocol mismatch')
    else:
        write(protocol_file, protocol)
    run_ids = [str(root/f'fold_{i}'/arm) for i, arms in enumerate(ORDER) for arm in arms]
    for i, arms in enumerate(ORDER):
        for arm in arms:
            dest = root/f'fold_{i}'/arm
            if (dest/'frozen.json').is_file():
                frozen = read(dest/'frozen.json')
                if digest(dest/'selected_model.joblib') != frozen['model_sha256']:
                    raise ValueError('Frozen model changed')
            else:
                config = dict(tabular_weights=str(args.weights.resolve()), ledger=str(args.ledger.resolve()), scope='task1',
                              cap_rmb=ARM_CAP, campaign_cap_rmb=CAMPAIGN_CAP, previous_run_ids=run_ids,
                              deferred_ids=historical[0]['legacy_unknown_ids'], level_assignment=arm)
                print('Starting', i, arm, flush=True)
                frozen = freeze_run(data[i][0], dest, config if arm!='original' else None)
            print('Frozen', i, arm, frozen['winner'], 'fits', frozen['fits'], 'valid', frozen['complete'], flush=True)
            if not frozen['complete']:
                raise RuntimeError(f'Incomplete paired study. Inspect {dest}/frozen.json before outer evaluation.')
    results = []
    total = {a:0 for a in ORDER[0]}
    for i, (_, features, labels) in enumerate(data):
        arms, preds = {}, {}
        for arm in ORDER[i]:
            dest = root/f'fold_{i}'/arm
            f = read(dest/'frozen.json')
            bundle = joblib.load(dest/'selected_model.joblib')
            if bundle['candidate'] != f['winner'] or digest(dest/'selected_model.joblib') != f['model_sha256']:
                raise ValueError('Invalid frozen winner at prediction')
            preds[arm] = predict_with_model(bundle['kind'], bundle['model'], features)
            if preds[arm].shape != labels.shape:
                raise ValueError('Prediction dimensions differ')
            correct = int(np.sum(preds[arm]==labels))
            total[arm] += correct
            before = f['trials'][:4]
            if arms and before != next(iter(arms.values()))['trials'][:4]:
                raise ValueError('The four original feedback observations differ across arms')
            arms[arm] = dict(**f, usage=usage(dest,args.ledger), correct=correct, accuracy=correct/len(labels),
                             decision=read(dest/'pretrained_decision.json') if arm!='original' else None)
        results.append(dict(rows=len(labels), arms=arms,
                            semantic_vs_original=paired(preds['original'], preds['semantic'], labels),
                            permuted_vs_original=paired(preds['original'], preds['permuted'], labels),
                            semantic_vs_permuted=paired(preds['permuted'], preds['semantic'], labels)))
        write(root/f'fold_{i}'/'outer_predictions.json', dict(rows=indices[i]['outer'], truth=labels.tolist(),
              **{a:y.tolist() for a,y in preds.items()}))
    n = sum(r['rows'] for r in results)
    diffs = [r['arms']['semantic']['accuracy']-r['arms']['original']['accuracy'] for r in results]
    cost = sum(a['usage']['course_rmb'] for r in results for a in r['arms'].values())
    if cost > CAMPAIGN_CAP:
        raise ValueError('Campaign allowance exceeded')
    promote = sum(d>0 for d in diffs)>=2 and (total['semantic']-total['original'])/n>=.002 and min(diffs)>=-.01 and total['semantic']>total['permuted']
    write(root/'results.json',dict(folds=results, rows=n, correct=total, accuracy={a:c/n for a,c in total.items()},
          promote=promote, course_rmb=cost))
    print('Complete', n, total, 'promote', promote, 'course RMB', cost, flush=True)


if __name__ == '__main__':
    main()
