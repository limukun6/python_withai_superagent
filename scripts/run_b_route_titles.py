"""Measure the effect of first-route parameter proposals on the two registered headline splits."""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from scripts import run_b1_ab_acceptance as rec, run_b1_05_pair_pilot as pilot
from scripts.run_b_a19_released_delivery import export_frozen
from scripts.run_b_c1_matrix import read_export
from superagent import ab_run
from superagent.ab_sentence import verified_directory
from superagent.data import load_development_task, load_prediction_features, load_training_task
from superagent.metrics import MetricSpec
from tools.ab_registered_rows import content_digest


SEEDS = ('45101','45102')
STABLE = '4d29aeb360e0a15f20011d4b98703bbcc4ebe332e11e7a40716b2d8919651ce3'
PLAN = 'docs/batches/B_ROUTE_02_03_released_plan.md'


def setup(out, weights):
    out.mkdir(parents=True,exist_ok=False)
    rec.offline(out)
    os.environ.update(SUPERAGENT_OFFLINE='1',SUPERAGENT_MINILM_WEIGHTS=str(weights.resolve()),
                      OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    os.environ.pop('SUPERAGENT_MINILM_CACHE_DIR',None)


def sources(control):
    versions = [{p.relative_to(root/'superagent').as_posix():ab_run.sha(p) for p in (root/'superagent').rglob('*.py')}
                for root in (control,rec.ROOT)]
    old,new = versions
    if rec.digest(old)!=STABLE or {k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}!={'ab_scheduler.py'}:
        raise ValueError('Two executable trees must differ only in the registered B scheduler')
    return dict(original=rec.digest(old),representative=rec.digest(new))


def stage_case(case, packet, entry):
    case.mkdir()
    for name,sha in [('train.zip',entry['files']['train.zip']),('inner.json',entry['inner_sha256'])]:
        if ab_run.sha(packet/name)!=sha:
            raise ValueError('Registered optimization input changed')
        os.link(packet/name,case/name)
    shutil.copytree(packet/'registered',case/'registered',copy_function=os.link)
    rec.write(case/'entry.json',dict(entry,name=case.name))


def worker(args):
    if rec.ROOT!=args.root or rec.digest(ab_run.sources())!=args.source:
        raise ValueError('Worker imported code from the wrong checkout')
    entry = rec.read(args.case/'entry.json')
    if args.phase=='train':
        if ab_run.sha(args.case/'inner.json')!=entry['inner_sha256']:
            raise ValueError('Registered rows changed')
        command = ('-m','tools.ab_registered_run','--split-manifest',args.case/'inner.json','--split-sha256',entry['inner_sha256'],
                   '--','run','--catalog-scope','full','--planner','local','--history-policy','strict')
        result = pilot.train_one(args.case,entry,args.source,launcher=command)
        manifest = rec.read(args.case/'run/run_manifest.json')
        if manifest['task_contract']!=entry['contract'] or list(manifest['capabilities'])!=entry['candidate_ids']:
            raise ValueError('Worker used a different task or ordered catalogue')
        rec.write(args.case/'training.json',result)
    else:
        if ab_run.sha(args.features)!=entry['features_sha256']:
            raise ValueError('Prediction rows changed')
        rec.write(args.case/'delivery.json',dict(csv_sha256=export_frozen(args.case,args.features,entry['rows'])))


def invoke(case,root,source,phase,features):
    env = dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,(root/'scripts/ab_acceptance_guard',root))))
    cmd = [sys.executable, str(Path(__file__).resolve()), 'worker', '--case',case,'--root',root,'--source',source,
           '--phase',phase,'--features',features]
    with (case/(phase+'_driver.log')).open('x') as log:
        subprocess.run(list(map(str,cmd)),cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1320)


def train(case,root,source,features,out):
    before = len(rec.trace(out))
    invoke(case,root,source,'train',features)
    job = rec.read(case/'training.json')
    actual = rec.trace(out)[before:]
    expected = [w['candidate'] for w in job['workers'] for _ in range(w['attempted_classifier_fits'])]
    if len(actual)!=job['fits'] or [row['candidate'] for row in actual]!=expected or not 1<job['fits']<=11:
        raise ValueError('Independent physical fits differ from scheduler ledger')
    return job


def paired_score(truth, metric, paths):
    flags = {}
    result = {}
    for arm,path in paths.items():
        pred = np.asarray(read_export(path,len(truth)),dtype=truth.dtype)
        flags[arm] = pred==truth
        result[arm] = dict(correct=int(flags[arm].sum()),rows=len(truth),scores=metric.evaluate(truth,pred),csv_sha256=ab_run.sha(path))
    a,b = flags['original'],flags['representative']
    return result, dict(gained=int((~a & b).sum()),lost=int((a & ~b).sum()),delta_accuracy_pp=100*(float(b.mean())-float(a.mean())))


def study(args):
    from scripts.replay_b_route_representative import pinned
    out = args.output.resolve()
    setup(out,args.weights)
    origin = args.previous.resolve()
    root = args.control.resolve()
    source = sources(root)
    previous_root = origin.parent.parent
    def previous(file):
        return pinned(previous_root,'a9cc24276610cd06c7fdacc0b66ec96743e2ebd2','b_contextual_preference',origin/file)
    prior,pre,frozen = (previous(f) for f in ('protocol.json','preflight.json','all_models_frozen.json'))
    if source['original']!=prior['production']['original'] or pilot.environment()!=prior['environment'] or verified_directory() is None:
        raise ValueError('Baseline runtime, model weights or dependencies changed')
    ref = rec.read(rec.ROOT/'research/ba19_released_delivery/preflight.json')['headlines']
    if ab_run.sha(args.staged)!=ref['source_staged_sha256']:
        raise ValueError('Wrong training-only source archive')
    rec.write(out/'protocol.json',dict(batch='B-ROUTE-02',sources=source,plan_sha256=ab_run.sha(rec.ROOT/PLAN),
              runner_sha256=ab_run.sha(Path(__file__)),registered_inputs_sha256=ab_run.sha(origin/'preflight.json'),
              environment=pilot.environment(),seeds=SEEDS,fit_limit=11,max_new_fits=22,wall_seconds=1200,worker_seconds=300,model_api_limit=0))
    jobs = {}
    try:
        for seed in SEEDS:
            entry = pre['entries'][seed+'_original']
            packet = origin/seed
            case = out/seed
            historical = origin/(seed+'_original')/'run'
            selected = rec.read(historical/'final_selection.json')
            if (ab_run.sha(historical/selected['model_file'])!=frozen[seed+'_original'] or not selected['refit']
                    or list(rec.read(historical/'run_manifest.json')['capabilities'])!=entry['candidate_ids']
                    or len(entry['candidate_ids'])!=23 or not rec.read(historical/'run_outcome.json')['full_refit_completed']):
                raise ValueError('Reference model or complete menu changed')
            stage_case(case,packet,entry)
            jobs[seed] = train(case,rec.ROOT,source['representative'],packet/'features.zip',out)
            rec.write(out/'checkpoint.json',jobs)
        rec.write(out/'all_models_frozen.json',{s:{'original':frozen[s+'_original'],'representative':j['model_sha256']} for s,j in jobs.items()})
        baseline = previous('results.json')['paths']
        task = load_training_task(args.staged)
        pairs = {}
        old_fits = 0
        for seed,job in jobs.items():
            case = out/seed
            packet = origin/seed
            old = origin/(seed+'_original')
            signed = baseline[old.name]
            raw = gzip.decompress((packet/'outer_ids.json.gz').read_bytes())
            details = pre['tasks'][seed]
            if rec.sha(raw)!=details['outer_ids_sha256']:
                raise ValueError('Held row identities changed')
            truth = task.y_train[json.loads(raw)['held']]
            dev = load_development_task(case/'train.zip')
            rows = load_prediction_features(packet/'features.zip',dev.kind,dev.spec)
            if content_digest(rows,truth)!=details['held_content_sha256']:
                raise ValueError('Held rows or label order changed')
            ledger = rec.read(old/'run/classifier_fits.json')
            if (signed['actions']!=rec.read(old/'run/actions.json') or signed['model_sha256']!=frozen[old.name]
                    or ledger['workers']!=signed['workers'] or ledger['attempted_classifier_fits']!=signed['fits']
                    or any(ab_run.sha(old/f'export_{i}/rows.csv')!=signed['csv_sha256'] for i in range(2))):
                raise ValueError('Signed comparison trajectory, model, fits or exports changed')
            old_fits += signed['fits']
            invoke(case,rec.ROOT,source['representative'],'deliver',packet/'features.zip')
            grade,delta = paired_score(truth,MetricSpec.for_task(dev),dict(original=old/'export_0/rows.csv',representative=case/'export_0/rows.csv'))
            if grade['original']['correct']!=signed['correct'] or grade['original']['scores']!=signed['scores']:
                raise ValueError('Inherited control did not reproduce its registered grade')
            pairs[seed] = dict(grade=grade,paired=delta,original_selected=signed['selected'],new_selected=job['selected'],
                               original_actions=signed['actions'],new_actions=job['actions'])
        fits = len(rec.trace(out))
        if fits!=sum(j['fits'] for j in jobs.values()) or fits>22 or sources(root)!=source:
            raise ValueError('Final source or fit accounting changed')
        rec.write(out/'results.json',dict(new_paths=jobs,pairs=pairs))
        rec.write(out/'completion.json',dict(status='completed',new_fits=fits,inherited_fits=old_fits,new_cli_exports=4,
                  verified_old_exports=4,new_model_api_requests=0,new_course_rmb=0,official_validation_test_or_H=False))
    except Exception as exc:
        rec.write(out/'completion.json',dict(status='failed',reason=str(exc),new_fits=len(rec.trace(out))))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode',required=True)
    parent = modes.add_parser('study')
    for name in ('control','previous','staged','weights','output'):
        parent.add_argument('--'+name,type=Path,required=True)
    child = modes.add_parser('worker')
    for name in ('case','root','features'):
        child.add_argument('--'+name,type=Path,required=True)
    child.add_argument('--source',required=True)
    child.add_argument('--phase',choices=('train','deliver'),required=True)
    args = parser.parse_args()
    if args.mode=='study':
        study(args)
    else:
        worker(args)
