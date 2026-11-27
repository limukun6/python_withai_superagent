"""Choose and confirm interventions by two distinct encoders using disjoint training groups."""
import argparse
import hashlib
from itertools import combinations
from pathlib import Path

import joblib
import numpy as np
from scipy.stats import binomtest

from scripts.compare_regrouped_portfolios import lock
from scripts.confirm_unused_training_groups import folds
from scripts.study_weighted_evidence import FOLDS, available
from scripts.unified_released_study import ROOT, read, require, sha
from superagent.autonomous_run import planner_observation
from superagent.data import TrainingTask
from superagent.experiment_policy import identity
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups

PLAN = ROOT / 'research/paired_encoder_search_1127/consensus_protocol.json'


def group_buckets(groups):
    def bucket(group):
        key = f'paired feedback consensus group v1/{group}'.encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') % 3
    return np.array([bucket(g) for g in groups])


def comparison(mask, incumbent, truth, groups):
    impact = np.where(incumbent == truth, -1, 1)[mask]
    unique, inverse = np.unique(groups[mask], return_inverse=True)
    by_group = np.bincount(inverse, weights=impact, minlength=len(unique))
    gains = int(np.sum(impact > 0))
    losses = int(np.sum(impact < 0))
    return dict(gains=gains, losses=losses, net=gains-losses,
                up=int(np.sum(by_group > 0)), down=int(np.sum(by_group < 0)))


def opposed(first, second, incumbent_probability):
    sign = first > .5
    change = (sign == (second > .5)) & (sign != (incumbent_probability > .5))
    margin = np.minimum(np.abs(first - .5), np.abs(second - .5)) - np.abs(incumbent_probability - .5)
    return change, margin


def change_mask(bank, rule, incumbent_probability):
    if rule is None:
        return np.zeros(len(incumbent_probability), dtype=bool)
    a, b = rule['members']
    mask, margin = opposed(bank[a], bank[b], incumbent_probability)
    threshold = rule['threshold']
    return mask if threshold is None else mask & (margin >= threshold)


def select(bank, incumbent_probability, truth, groups, encoders, configs, incumbents):
    key = lambda n: configs[n]
    parent_families = {encoders[n] for n in incumbents}
    challengers = sorted((n for n in bank if encoders[n] not in parent_families), key=key)
    best_order, winner = None, None
    for a, b in combinations(challengers, 2):
        if encoders[a] == encoders[b]:
            continue
        opposed_mask, margin = opposed(bank[a], bank[b], incumbent_probability)
        if not opposed_mask.any():
            continue
        eligible_margins = margin[opposed_mask]
        thresholds = [None, float(np.median(eligible_margins)), float(np.quantile(eligible_margins, .75))]
        for level, threshold in enumerate(thresholds):
            mask = opposed_mask if threshold is None else opposed_mask & (margin >= threshold)
            measure = comparison(mask, incumbent_probability > .5, truth, groups)
            if measure['up'] + measure['down'] < 5:
                continue
            order = (-measure['net'], measure['down']-measure['up'], int(mask.sum()), level, key(a), key(b))
            if best_order is None or order < best_order:
                best_order = order
                winner = dict(members=[a,b], level=level, threshold=threshold, learn=measure)
    return winner


def examine(out, fold_name, bank, truth, groups, families, configs, incumbents):
    p = np.mean([bank[n] for n in incumbents], axis=0)
    buckets = group_buckets(groups)
    require(set(buckets) == {0,1,2}, 'Missing component role')
    rotations = []
    combined = {k:np.zeros(len(truth), bool) for k in ('confirmed','nominee','always')}
    for r in range(3):
        learn, confirm, test = [buckets == (r+i)%3 for i in range(3)]
        winner = select({n:v[learn] for n,v in bank.items()},p[learn],truth[learn],groups[learn],
                        families,configs,incumbents)
        lock(out/f'{fold_name}_{r}_learn.json',dict(role=r,rows=int(learn.sum()),rule=winner))
        mask = change_mask(bank, winner, p)
        confirmation = comparison(mask[confirm],p[confirm]>.5,truth[confirm],groups[confirm])
        decisive = confirmation['up'] + confirmation['down']
        pvalue = float(binomtest(confirmation['up'],decisive,.5,alternative='greater').pvalue) if decisive else 1.
        accept = (winner is not None and winner['learn']['net']>0 and confirmation['net']>0
                  and decisive>=5 and pvalue<=.10)
        decision = dict(accept=accept,sign_p=pvalue,confirmation=confirmation,rows=int(confirm.sum()))
        lock(out/f'{fold_name}_{r}_confirm.json', decision)
        always = change_mask(bank,dict(winner,threshold=None) if winner else None,p)
        for name,changes in [('confirmed',mask & accept),('nominee',mask),('always',always)]:
            combined[name][test] = changes[test]
        rotations.append(dict(rotation=r, rule=winner, **decision, evaluation_rows=int(test.sum()),
                              evaluation=comparison((mask & accept)[test],p[test]>.5,truth[test],groups[test])))
    measures = {k:comparison(changes,p>.5,truth,groups) for k,changes in combined.items()}
    return dict(fold=fold_name,rows=len(truth),baseline=int(np.sum((p>.5)==truth)),
                results=measures, rotations=rotations)


def aligned_training_predictions(root,out,plan):
    agent = root / 'raw/race/run/agent'
    data = joblib.load(agent/'refit_data.joblib')
    full_labels = np.asarray(data['y_train'])
    all_groups = sample_groups(TrainingTask(data['kind'],data['X_train'],full_labels),'shared_text')
    seal = root/'auto-raw/auto_portfolio/sealed.json'
    inc = read(seal)['members']
    require(len(inc)==2 and read(seal)['independent_reload'], 'Invalid frozen incumbent')
    original = available(root)
    first_three = [np.asarray(read(agent/(s+'_split.json'))['held']) for s in FOLDS]
    last_two = [held for _,held in folds(data,agent)]
    all_held = first_three + last_two
    require(np.array_equal(np.sort(np.concatenate(all_held)),np.arange(len(full_labels))),'Incomplete base OOF')
    slots = []
    for i,held in enumerate(all_held):
        if i < 3:
            tag = FOLDS[i]
            payload = agent / (tag+'.joblib')
            require(read(agent/(tag+'_split.json'))['payload_sha256'] == sha(payload),'Changed base payload')
            paths = original[tag]
        else:
            tag = f'fold_{i+1}'
            directory = root/'unused-groups'/tag
            payload = directory/'training.joblib'
            stamp = read(directory/'rows.json')
            require(stamp['held_sha256']==fingerprint(held) and stamp['payload_sha256']==sha(payload),'Wrong held samples')
            paths = {}
            for path in directory.glob('*/trial.joblib'):
                receipt = read(path.parent/'receipt.json')
                require(receipt['trial_sha256']==sha(path) and receipt['payload_sha256']==stamp['payload_sha256'],
                        'Invalid component receipt')
                paths[receipt['member']] = path
        slots.append((tag, payload, paths, held))
    members = set.intersection(*(set(paths) for _,_,paths,_ in slots))
    require(set(inc)<=members, 'Missing incumbent probabilities')
    lock(out/'inputs.json',dict(protocol=sha(plan),incumbent=sha(seal),members=sorted(members),folds=[
        dict(name=tag,held=fingerprint(held),payload=sha(payload),trials={n:sha(paths[n]) for n in sorted(members)})
        for tag,payload,paths,held in slots]))
    expected_configs = None
    for tag,payload,paths,held in slots:
        context = joblib.load(payload)
        metric = MetricSpec.from_payload(context)
        labels = np.asarray(context['y_search'])
        require(metric.name=='accuracy' and len(metric.labels)==2 and np.array_equal(labels,full_labels[held]),
                'Unexpected metric, score labels or row order')
        groups = all_groups[held]
        bank,families,configs = {},{},{}
        payload_key = sha(payload)
        for n in sorted(members):
            trial = joblib.load(paths[n])
            config = trial['metrics']['resolved_configuration']
            require(config['id']==n,'Mismatched member')
            params = config['params']
            if not params.get('encoder'):
                continue
            obs = planner_observation(trial,trial['metrics'],dict(sha256=payload_key,labels=labels),metric)
            probs = np.asarray(obs['probabilities'],dtype=float)
            require(probs.shape==(len(held),2) and np.isfinite(probs).all()
                    and np.allclose(probs.sum(axis=1),1) and np.all((0<=probs)&(probs<=1)),'Invalid scores')
            bank[n] = probs[:,1]
            families[n] = params['encoder']
            configs[n] = identity(params)
        require(expected_configs is None or expected_configs==configs,'A component configuration changed')
        expected_configs = configs
        yield tag,bank,labels==metric.labels[1],groups,families,configs,inc


def run(root):
    out=root/'consensus-feedback'
    out.mkdir(exist_ok=True)
    records=[examine(out,*item) for item in aligned_training_predictions(root,out,PLAN)]
    total=sum(r['rows'] for r in records)
    results = {arm:{k:sum(r['results'][arm][k] for r in records) for k in ('gains','losses','net')}
               for arm in ('confirmed','nominee','always')}
    net = [r['results']['confirmed']['net'] for r in records]
    advance = (results['confirmed']['net']/total>=.003 and sum(x>0 for x in net)>=3
               and all(x/r['rows']>=-.003 for x,r in zip(net,records)))
    result = dict(protocol_sha256=sha(PLAN),inputs_sha256=sha(out/'inputs.json'),rows=total,
                  baseline=sum(r['baseline'] for r in records),results=results,folds=records,advance=advance,
                  accepted=sum(x['accept'] for r in records for x in r['rotations']),supervised_rule_searches=15,
                  base_classifier_fits=0,model_api_requests=0,designated_validation_opened=False,
                  formal_test_opened=False,H2_opened=False)
    lock(out/'results.json',result)
    print(result['rows'],result['baseline'],results,result['accepted'],net,advance,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=lambda s:Path(s).resolve(),required=True)
    run(parser.parse_args().root)
