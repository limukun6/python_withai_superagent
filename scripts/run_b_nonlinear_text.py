"""Compare one extra executable frozen-text method on registered headline or SMS data."""
import argparse
import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess

import joblib
import numpy as np

from scripts import run_b1_ab_acceptance as rec, run_b1_05_pair_pilot as pilot
from scripts.run_b_a19_released_delivery import export_frozen
from scripts.run_b_c1_matrix import read_export
from superagent import ab_run
from superagent.ab_methods import TEXT_KERNEL
from superagent.ab_sentence import verified_directory
from superagent.contracts import TaskSpec
from superagent.data import load_development_task, load_prediction_features, load_training_task
from superagent.metrics import MetricSpec
from tools.ab_registered_rows import content_digest


ARCHIVES = {
    'headlines':('a9cc24276610cd06c7fdacc0b66ec96743e2ebd2','b_contextual_preference',('45101','45102')),
    'sms':('00935216e85790dbb0232a00d11c3a902a5eb1d3','b_cold_probe_prospective',('sms',)),
}
STABLE = '4d29aeb360e0a15f20011d4b98703bbcc4ebe332e11e7a40716b2d8919651ce3'


def signed(origin,commit,collection,file):
    data = subprocess.check_output(['git','show',commit+':research/'+collection+'/'+file],cwd=rec.ROOT)
    if rec.sha(data)!=ab_run.sha(origin/file):
        raise ValueError('External registration or signed control changed: '+file)
    return json.loads(data)


def identity(control):
    old,new = ({p.relative_to(root/'superagent').as_posix():ab_run.sha(p) for p in (root/'superagent').rglob('*.py')}
               for root in (control,rec.ROOT))
    if (rec.digest(old)!=STABLE or {k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
            != {'ab_sentence.py','ab_methods.py','ab_catalog.py'}):
        raise ValueError('Registered comparison changed additional production code')
    return {'original':STABLE,'extended':rec.digest(new)}


def prepare(case,packet,past,entry,frozen):
    dev = load_development_task(packet/'train.zip')
    caps = ab_run.registered_capabilities(TaskSpec.from_task(dev).record(),len(dev.y_train),'full')
    old = rec.read(past/'run/run_manifest.json')
    before = old['capabilities']
    chosen = rec.read(past/'run/final_selection.json')
    if (dev.X_val is not None or dev.y_val is not None or rec.digest(old['source_files'])!=STABLE
            or old['task_contract']!=entry['contract'] or old['task_contract']!=TaskSpec.from_task(dev).record()
            or old['time_limit_seconds']!=1200 or old['trial_timeout_seconds']!=300 or old['fit_limit']!=11
            or list(before)!=entry['candidate_ids'] or rec.digest(before)!=entry['catalog_sha256']
            or list(caps)!=[*before,TEXT_KERNEL] or any(caps[k]!=v for k,v in before.items())
            or not chosen['refit'] or chosen['model_sha256']!=frozen or ab_run.sha(past/'run'/chosen['model_file'])!=frozen):
        raise ValueError('Old sources, common task, registered model or catalogue membership differs')
    case.mkdir()
    for name,sha in [('train.zip',entry['files']['train.zip']),('inner.json',entry['inner_sha256'])]:
        if ab_run.sha(packet/name)!=sha:
            raise ValueError('Registered optimization packet or internal rows changed')
        os.link(packet/name,case/name)
    shutil.copytree(packet/'registered',case/'registered',copy_function=os.link)
    updated = dict(entry,name=case.name,catalog_sha256=rec.digest(caps),candidate_ids=list(caps))
    rec.write(case/'entry.json',updated)
    return updated


def pair_grade(truth,task,paths):
    flags = {}
    grades = {}
    for arm,path in paths.items():
        pred = np.asarray(read_export(path,len(truth)),dtype=truth.dtype)
        flags[arm] = pred==truth
        grades[arm] = dict(correct=int(flags[arm].sum()),rows=len(truth),scores=MetricSpec.for_task(task).evaluate(truth,pred),
                           csv_sha256=ab_run.sha(path))
    left,right = flags.values()
    return grades,dict(gained=int((~left & right).sum()),lost=int((left & ~right).sum()),
                       delta_primary_pp=100*(grades['extended']['scores']['score']-grades['original']['scores']['score']))


def study(args):
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    rec.offline(out)
    os.environ.update(SUPERAGENT_OFFLINE='1',SUPERAGENT_MINILM_WEIGHTS=str(args.weights.resolve()),
                      OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    os.environ.pop('SUPERAGENT_MINILM_CACHE_DIR',None)
    origin,control = args.previous.resolve(),args.control.resolve()
    source = identity(control)
    commit,collection,keys = ARCHIVES[args.task]
    pre,proto,frozen = (signed(origin,commit,collection,f) for f in ('preflight.json','protocol.json','all_models_frozen.json'))
    headline = args.task=='headlines'
    if (verified_directory() is None or (headline and (pilot.environment()!=proto['environment'] or args.staged is None))
            or (not headline and proto['original_production']!=STABLE)):
        raise ValueError('Weights, prior runtime or training-only packet changed')
    rec.write(out/'protocol.json',dict(batch='B-TEXT-KERNEL-02' if headline else 'B-TEXT-KERNEL-03',
              sources=source,driver_sha256=ab_run.sha(Path(__file__)),plan_sha256=ab_run.sha(rec.ROOT/'docs/batches/B_TEXT_KERNEL_01_03_plan.md'),
              registered_sources=commit,prior_preflight_sha256=ab_run.sha(origin/'preflight.json'),
              frozen_keys=keys,fit_limit=11,max_new_fits=11*len(keys),wall_seconds=1200,worker_seconds=300,api_limit=0))
    jobs,entries = {},{}
    try:
        if headline and ab_run.sha(args.staged)!=rec.read(rec.ROOT/'research/ba19_released_delivery/preflight.json')['headlines']['source_staged_sha256']:
            raise ValueError('Wrong training-only headline source')
        for k in keys:
            old_key = k+'_original' if headline else k
            entry = pre['entries'][old_key] if headline else pre[k]
            entries[k] = prepare(out/k,origin/k,origin/old_key,entry,frozen[old_key])
        rec.write(out/'preflight.json',entries)
        for k,entry in entries.items():
            case = out/k
            launch = ('-m','tools.ab_registered_run','--split-manifest',case/'inner.json','--split-sha256',entry['inner_sha256'],'--',
                      'run','--catalog-scope','full','--planner','local','--history-policy','strict')
            start = len(rec.trace(out))
            job = pilot.train_one(case,entry,source['extended'],launcher=launch)
            observed = rec.trace(out)[start:]
            calls = [w['candidate'] for w in job['workers'] for _ in range(w['attempted_classifier_fits'])]
            actual = rec.read(case/'run/run_manifest.json')
            if ([r['candidate'] for r in observed]!=calls or len(observed)!=job['fits'] or actual['task_contract']!=entry['contract']
                    or list(actual['capabilities'])!=entry['candidate_ids']):
                raise ValueError('Actual fits, task, or candidate order changed')
            history = rec.read(case/'run/history.json')
            job['new_candidate_trials'] = [
                dict(trial=r['trial'],split=r['split_id'],search_score=r['search_score'],worker_seconds=r['worker_wall_seconds'],
                     usage=r['training_metadata']['usage']) for r in history if r['candidate']==TEXT_KERNEL and r['status']=='completed'
            ]
            job['new_candidate_selected'] = job['selected']==TEXT_KERNEL
            jobs[k] = job
            rec.write(out/'checkpoint.json',jobs)
        rec.write(out/'all_models_frozen.json',{k:dict(extended=j['model_sha256'],original=frozen[k+'_original' if headline else k])
                                                 for k,j in jobs.items()})
        reference = signed(origin,commit,collection,'results.json')['paths']
        pairs = {}
        old_fits = 0
        source_task = load_training_task(args.staged) if headline else None
        for k,entry in entries.items():
            packet = origin/k
            old_key = k+'_original' if headline else k
            prior = origin/old_key
            job = reference[old_key]
            old_ledger = rec.read(prior/'run/classifier_fits.json')
            if (job['actions']!=rec.read(prior/'run/actions.json') or job['model_sha256']!=frozen[old_key]
                    or old_ledger['attempted_classifier_fits']!=job['fits'] or old_ledger['completed_classifier_fits']!=job['fits']
                    or any(ab_run.sha(prior/f'export_{i}/rows.csv')!=job['csv_sha256'] for i in range(2))):
                raise ValueError('Frozen baseline action, fit budget or predictions changed')
            old_fits += job['fits']
            dev = load_development_task(packet/'train.zip')
            query = packet/'features.zip'
            if ab_run.sha(query)!=(entry['features_sha256'] if headline else entry['files']['features.zip']):
                raise ValueError('Frozen delivery packet changed')
            values = load_prediction_features(query,dev.kind,dev.spec)
            if headline:
                raw = gzip.decompress((packet/'outer_ids.json.gz').read_bytes())
                if rec.sha(raw)!=pre['tasks'][k]['outer_ids_sha256']:
                    raise ValueError('Headline row identity changed')
                truth = source_task.y_train[json.loads(raw)['held']]
                registered = pre['tasks'][k]['held_content_sha256']
            else:
                if ab_run.sha(packet/'holdout.joblib')!=entry['files']['holdout.joblib']:
                    raise ValueError('Public SMS labels changed')
                held = joblib.load(packet/'holdout.joblib')
                truth = held['y']
                registered = content_digest(held['X'],truth)
            if content_digest(values,truth)!=registered or len(values)!=entry['rows']:
                raise ValueError('Delivery rows and graded truth do not line up')
            digest = export_frozen(out/k,query,len(truth))
            grades,delta = pair_grade(truth,dev,dict(original=prior/'export_0/rows.csv',extended=out/k/'export_0/rows.csv'))
            if grades['original']['scores']!=job['scores'] or grades['extended']['csv_sha256']!=digest:
                raise ValueError('Prior grading or new relocated predictions changed')
            pairs[k] = dict(grades=grades,paired=delta,prior_selected=job['selected'],extended_selected=jobs[k]['selected'])
        fits = len(rec.trace(out))
        if fits>11*len(keys) or fits!=sum(j['fits'] for j in jobs.values()) or identity(control)!=source:
            raise ValueError('Final supervised fit ledger or executable sources changed')
        rec.write(out/'results.json',dict(new_paths=jobs,pairs=pairs))
        rec.write(out/'completion.json',dict(status='completed',new_fits=fits,inherited_fits=old_fits,new_cli_exports=2*len(keys),
                  verified_old_exports=2*len(keys),new_model_api_requests=0,new_course_rmb=0,official_validation_test_or_H=False))
    except Exception as exc:
        rec.write(out/'completion.json',dict(status='failed',reason=str(exc),physical_fit_starts=len(rec.trace(out))))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=ARCHIVES)
    for name in ('previous','control','weights','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--staged',type=Path)
    study(parser.parse_args())
