"""Test a primary-selected disagreement rule on the already locked per-row training predictions."""
import argparse
import math
from pathlib import Path

import joblib
import numpy as np

from scripts.study_task1_cell_encoders import CASES, FOLDS
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.autonomous_run import planner_observation
from superagent.cell_encoder_review import passes, score
from superagent.metrics import MetricSpec

PLAN = ROOT/'research/task1_feedback_extension_1127/cell_disagreement_gate.json'


def margin(matrix):
    largest = np.partition(matrix, -2, axis=1)[:, -2:]
    return largest[:, 1]-largest[:, 0]


def route(metrics, truth, prior, alternative, rule):
    labels = np.asarray(metrics.labels)
    old, new = prior.argmax(1), alternative.argmax(1)
    use = (old != new) & (margin(prior) <= rule['threshold'])
    if rule['higher_expert_margin']:
        use &= margin(alternative) > margin(prior)
    blended = prior.copy()
    blended[use] = (1-rule['share'])*prior[use] + rule['share']*alternative[use]
    baseline, elected = labels[old]==truth, labels[blended.argmax(1)]==truth
    return dict(**score(metrics,truth,blended), switched=int(np.sum(blended.argmax(1)!=old)),
                fixed=int((elected & ~baseline).sum()), lost=int((baseline & ~elected).sum()))


def row_matrices(root, key, report, fold, keys):
    native = root/'cell_portfolio_1128'/key/'portfolio/run/agent'
    original = root/CASES[key]/'replicated/run/agent' if 'original' in read(native.parent.parent/'launch.json') else native
    members = read(native/'selection.json')['choice']['members']
    trace = {r['candidate']:r for r in read(original/'history.json') if r['status']=='completed' and r['split_id']==fold}
    path = native/(fold+'.joblib')
    truth = joblib.load(path)['y_search']
    metric = MetricSpec.from_payload(dict(metric_spec=read(native/'task_contract.json')['metric']))
    digest = sha(path)
    require(digest==report['partitions'][fold]['hash'], 'Wrong training partition')
    def matrix(location, expected, candidate, seed=None):
        data = joblib.load(location)
        facts = data['metrics']
        require(facts['resolved_configuration']['id']==candidate and (seed is None or
                facts['resolved_configuration'].get('training_random_seed')==seed), 'Wrong model or seed')
        return planner_observation(data,facts,dict(sha256=expected,labels=truth),metric)['probabilities']
    repeated = read(native/'seed_replication.json')
    parts = []
    for i, name in enumerate(members):
        matrices = [matrix(original/f"trial_{trace[name]['trial']}.joblib",digest,name)]
        if fold!='primary' and repeated['action']=='adopt' and name in repeated['eligible']:
            for seed in repeated['seeds']:
                start = native/f'cell_{fold}_seed_{seed}.joblib'
                state = native/f'cell_{fold}_reference_{i}_{seed}.joblib'
                if not state.exists():
                    require(repeated['pilot_split']==fold and repeated['seeds'][0]==seed, 'Missing matching seed')
                    state = native/f'seed_pilot_{i}.joblib'
                matrices.append(matrix(state,sha(start),name,seed))
        parts.append(np.mean(matrices,axis=0))
    alternate = [matrix(native/f'cell_{fold}_{k}.joblib',digest,'catboost_cell_'+k) for k in keys]
    return metric, truth, np.mean(parts,axis=0), np.mean(alternate,axis=0)


def review(root, key):
    native=root/'cell_portfolio_1128'/key/'portfolio/run/agent'
    report=read(native/'cell_encoder_review.json')
    keys=report['nominee']['encoders']
    if not keys:
        return dict(action='skip',reason='Primary preferred the incumbent; no confirmed encoder to route')
    metric,y,base,side=row_matrices(root,key,report,'primary',keys)
    initial=score(metric,y,base)
    options=[]
    for q in (.25,.5):
        for stronger in (False,True):
            for share in (.5,.75,1.):
                rule=dict(quantile=q,threshold=float(np.quantile(margin(base),q)),higher_expert_margin=stronger,share=share)
                observation=route(metric,y,base,side,rule)
                if observation['switched'] >= max(10,math.ceil(.01*len(y))):
                    options.append((rule,observation))
    winner=min(options,key=lambda p:(-p[1]['score'],p[1]['brier'],p[1]['switched'])) if options else None
    lock = root/'cell_portfolio_1128'/key/'route_lock.json'
    accepted=winner is not None and (-winner[1]['score'],winner[1]['brier']) < (-initial['score'],initial['brier'])
    election=dict(rule=winner[0] if accepted else None,primary=winner[1] if accepted else initial,encoder_keys=keys,
                  initial=initial,protocol=sha(PLAN),parent_decision=sha(native/'cell_encoder_review.json'))
    save(lock,election)
    checks=[]
    for fold in FOLDS[1:]:
        metric,y,base,side=row_matrices(root,key,report,fold,keys)
        prior=score(metric,y,base)
        expert=score(metric,y,side)
        a=np.asarray(metric.labels)[base.argmax(1)]==y
        b=np.asarray(metric.labels)[side.argmax(1)]==y
        checks.append(dict(fold=fold,rows=len(y),incumbent=prior,expert=expert,
                           correctable=int((b & ~a).sum()),at_risk=int((a & ~b).sum()),
                           challenger=route(metric,y,base,side,election['rule']) if accepted else dict(prior,switched=0,fixed=0,lost=0)))
    return dict(action='qualify' if accepted and passes(checks) else 'retain',lock=sha(lock),**election,confirmations=checks)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=lambda s:Path(s).resolve(),required=True)
    root=p.parse_args().root
    panels={key:review(root,key) for key in CASES}
    passing=[name for name,value in panels.items() if value['action']=='qualify']
    value=dict(panels=panels,protocol=sha(PLAN),fits=0,api_calls=0,uses_outer_data=False,
               advances=all(any(k.startswith(tag) for k in passing) for tag in ('novel','ordinary')))
    save(root/'cell_portfolio_1128/conditional_results.json',value)
    print({k:v['action'] for k,v in panels.items()},'advance',value['advances'])
