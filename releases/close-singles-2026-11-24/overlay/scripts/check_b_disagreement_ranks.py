"""Replay a fixed label-free confidence rank against the frozen policies on three previously used development panels."""
import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

import joblib
import numpy as np

from scripts.run_b_autonomous_close_singles import ROOT, env, read, require, seal
from superagent import ab_run
from superagent.ab_complement import canonical_probabilities, split_columns
from superagent.ab_scheduler import selection_evidence
from superagent.data import load_development_task, load_prediction_features
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups


PLAN = ROOT/'docs/batches/B_disagreement_ranks_plan_20261124_zh.md'


def ranks(opening, values):
    reference = np.sort(np.abs(opening-.5))
    strength = np.abs(values-.5)
    return (np.searchsorted(reference,strength,'left') + np.searchsorted(reference,strength,'right'))/(2*len(reference))


def labels(p, first):
    return ((p>.5) | ((p==.5)&first)).astype(int)


def prepare(key, archive, run, frozen, out, cold=None):
    task = load_development_task(archive)
    manifest = read(run/'run_manifest.json')
    history = read(run/'actions.json')
    state = read(run/f'state_{len(history)}.json')
    check = read(run/'selection.json')['confirmation']['b_scheduler_review']
    pair = [a['candidate_id'] for a in history if a['action']=='confirm']
    require(len(pair)==len(set(pair))==2 and ab_run.sha(archive)==manifest['task_archive_sha256'] and
            task.y_val is None and ab_run.sha(run/'selected_model.joblib')==frozen, 'Input, model or pair differs')
    with tempfile.TemporaryDirectory() as folder:
        metric = MetricSpec.for_task(task)
        trusted, allowed = ab_run.freeze_partitions(task, Path(folder), manifest['seed'], metric, preference_vote=True)
        for s in allowed:
            require(ab_run.sha(run/(s+'.joblib'))==allowed[s]['evaluation_payload_sha256'], 'Registered worker data changed')
        selection_evidence(state,trusted_partitions=trusted)
        _, _, opening = split_columns(state,trusted,'primary',metric)
        if cold is None:
            _, first, second = split_columns(state,trusted,'confirmation_1',metric)
            previous = set(trusted['primary']['groups'])
            mask = np.array([g not in previous for g in trusted['confirmation_1']['groups']])
            ids = np.asarray(trusted['confirmation_1']['row_ids'])[mask]
            group = np.asarray(trusted['confirmation_1']['groups'])[mask]
            relative = [second[n][1][mask] for n in pair]
            hard = [second[n][0][mask] for n in pair]
            require(len(check['members'])==1 and check['members'][0] in pair,'Expected the delivered single')
            baseline = hard[pair.index(check['members'][0])]
            first = first[mask]
        else:
            bundle = joblib.load(run/'selected_model.joblib')['model']
            source = load_prediction_features(cold/'features.zip','pair_text',read(run/'task_spec.json'))
            require(pair==check['members'] and len(bundle.models)==2,'Cold model member order differs')
            first = np.array([a<b for a,b in source])
            relative = [canonical_probabilities(m,source)[1][:,1] for m in bundle.models]
            hard = [labels(v,first) for v in relative]
            with (cold/'control/export_0/rows.csv').open(newline='') as f:
                baseline = np.array([int(row['prediction']) for row in csv.DictReader(f)])
            require(np.array_equal(labels(np.mean(relative,axis=0),first),baseline),'Native export differs')
            ids=np.arange(len(first))
            group=np.empty(0,dtype=int)  # Group identifiers are held with the frozen evaluator for this panel.
    index = [ranks(opening[n][1],v) for n,v in zip(pair,relative,strict=True)]
    voted = labels(np.mean(relative,axis=0),first)
    proposed = np.where(index[0]>index[1],hard[0],np.where(index[1]>index[0],hard[1],baseline))
    require(np.all((hard[0]!=hard[1]) | (proposed==hard[0]) | (index[0]==index[1])),'Invalid agreement routing')
    destination=out/(key+'.npz')
    with destination.open('xb') as f:
        np.savez_compressed(f,baseline=baseline,equal=voted,proposed=proposed,ids=ids,groups=group,disagree=hard[0]!=hard[1])
    return dict(archive=ab_run.sha(archive), frozen_model=frozen, actions=ab_run.sha(run/'actions.json'), members=pair,
                baseline_members=check['members'], opening_rows=len(opening[pair[0]][1]), rows=len(ids), prediction_artifact=ab_run.sha(destination))


def run(a):
    os.environ.update(env(ROOT,a.weights,a.output/'unused'))
    from scripts import a38_predict_guard
    a.output.mkdir(parents=True,exist_ok=False)
    cold_ref=read(a.cold/'models_frozen.json')['arms']['control']['model']
    full_ref=read(a.full/'model_frozen.json')['model']
    movie_ref=read(a.movie/'model_frozen.json')['model']
    seal(a.output/'protocol.json',dict(plan=ab_run.sha(PLAN),runner=ab_run.sha(Path(__file__)),production=ab_run.sources(),
         cold_receipt=ab_run.sha(a.cold/'models_frozen.json'),full_receipt=ab_run.sha(a.full/'model_frozen.json'),
         movie_receipt=ab_run.sha(a.movie/'model_frozen.json'),strategy='within-model primary confidence percentile',fit_cap=0,api_cap=0))
    runs = {
        'cold_titles':prepare('cold_titles',a.cold/'anonymous.zip',a.cold/'control/agent',cold_ref,a.output,a.cold),
        'full_titles':prepare('full_titles',a.original/'anonymous.zip',a.full/'agent',full_ref,a.output),
        'movie':prepare('movie',a.movie/'anonymous.zip',a.movie/'autonomous',movie_ref,a.output)}
    seal(a.output/'predictions_frozen.json',dict(protocol=ab_run.sha(a.output/'protocol.json'),panels=runs))
    results, title_groups = {}, {}
    for key, record in runs.items():
        target=a.output/(key+'.npz')
        require(record['prediction_artifact']==ab_run.sha(target),'Sealed predictions changed')
        with np.load(target,allow_pickle=False) as f:
            pre=dict(f)
        if key=='cold_titles':
            spec=read(a.cold/'protocol.json')
            require(spec['scoring']==ab_run.sha(a.cold/'score_only.npz'),'Sealed scoring changed')
            with np.load(a.cold/'score_only.npz',allow_pickle=False) as f:
                y,group=f['y'],f['groups']
                require(fingerprint(f['ids'])==spec['held_ids'],'Unknown cold order')
        else:
            task=load_development_task((a.original if key=='full_titles' else a.movie)/'anonymous.zip')
            y,group=task.y_train[pre['ids']],pre['groups']
            require(np.array_equal(sample_groups(task,'shared_text')[pre['ids']],group),'Unknown confirmation groups')
        if key!='movie':
            title_groups[key]=set(group.tolist())
        hits={k:pre[k]==y for k in ('baseline','equal','proposed')}
        fixed=hits['baseline']
        gained=hits['proposed'] & ~fixed
        lost=fixed & ~hits['proposed']
        _, ix=np.unique(group,return_inverse=True)
        numerator=np.bincount(ix,weights=gained.astype(int)-lost.astype(int))
        sizes=np.bincount(ix)
        rng=np.random.default_rng(29013)
        boot=[]
        for _ in range(100):
            draw=rng.integers(0,len(sizes),size=(50,len(sizes)))
            boot.extend(100*numerator[draw].sum(axis=1)/sizes[draw].sum(axis=1))
        gap=int(gained.sum()-lost.sum())
        results[key]=dict(rows=len(y), groups=len(sizes), disagree=int(pre['disagree'].sum()),
                          correct={k:int(v.sum()) for k,v in hits.items()},gained=int(gained.sum()),lost=int(lost.sum()),
                          net=gap,delta_pp=100*gap/len(y),paired_group_95_pp=np.quantile(boot,[.025,.975]).tolist())
        by_group=np.stack([np.bincount(ix,weights=(~fixed&~hits['proposed']).astype(int)),np.bincount(ix,weights=lost),
                           np.bincount(ix,weights=gained),np.bincount(ix,weights=fixed&hits['proposed'])],axis=1).astype(int)
        seal(a.output/(key+'_groups.json'),by_group.tolist())
    cold=results['cold_titles']
    passes=sum(p['net']>0 for p in results.values())>=2 and all(p['net']>=0 for p in results.values()) and cold['delta_pp']>=.5 and cold['paired_group_95_pp'][0]>0
    seal(a.output/'results.json',dict(frozen=ab_run.sha(a.output/'predictions_frozen.json'),panels=results,
         shared_group_count_between_title_panels=len(title_groups['cold_titles']&title_groups['full_titles']),
         passes_gate=bool(passes),new_fits=0,model_api_calls=0,course_validation_read=False,formal_test_read=False))
    seal(a.output/'completion.json',{k:ab_run.sha(a.output/(k+'.json')) for k in
         ('protocol','predictions_frozen','cold_titles_groups','full_titles_groups','movie_groups','results')})
    print(json.dumps(read(a.output/'results.json')))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cold','full','original','movie','weights','output'):
        parser.add_argument('--'+name,type=lambda s:Path(s).resolve(),required=True)
    run(parser.parse_args())
