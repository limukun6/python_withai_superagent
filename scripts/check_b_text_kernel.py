"""Exercise the new downstream feature map, its fit ledger, fresh reloads and old-model compatibility."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import shutil
from unittest.mock import patch

import joblib
import numpy as np

from scripts import run_b1_ab_acceptance as rec, run_b1_05_pair_pilot as pilot
from superagent import ab_run
from superagent.ab_methods import TEXT_KERNEL
from superagent.ab_sentence import verified_directory
from superagent.contracts import TaskSpec
from superagent.data import load_development_task
from superagent.metrics import MetricSpec
from superagent.models import probability_with_model


def reload_model(args):
    model = joblib.load(args.model)['model']
    rows = joblib.load(args.queries)
    classes,probs = probability_with_model(args.kind,model,rows)
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1),1):
        raise ValueError('Invalid fresh-process probabilities')
    report = dict(rows=len(rows),classes=classes.tolist(),probability_sha256=rec.sha(probs.tobytes()))
    if args.kind=='pair_text':
        _,reverse = probability_with_model(args.kind,model,[(r,l) for l,r in rows])
        report['max_swap_probability_error'] = float(np.max(abs(reverse-probs[:,::-1])))
        if report['max_swap_probability_error']>1e-5:
            raise ValueError('Swapping two non-identical inputs changed preference probabilities unexpectedly')
    rec.write(args.output,report)


def check(args):
    out = args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    rec.offline(out)
    os.environ.update(SUPERAGENT_OFFLINE='1',SUPERAGENT_MINILM_WEIGHTS=str(args.weights.resolve()),
                      OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    os.environ.pop('SUPERAGENT_MINILM_CACHE_DIR',None)
    sms = args.sms.resolve()
    headline = args.headlines.resolve()
    rec.write(out/'protocol.json',dict(batch='B-TEXT-KERNEL-01',maximum_fits=2,api_limit=0,sampling_seed=88017,
              plan_sha256=ab_run.sha(rec.ROOT/'docs/batches/B_TEXT_KERNEL_01_03_plan.md'),
              driver_sha256=ab_run.sha(Path(__file__)),production=rec.digest(ab_run.sources())))
    try:
        sms_entry = rec.read(sms/'entry.json')
        if (ab_run.sha(sms.parent/'preflight.json')!='630a11762c8154429d5b82d67ae97f1b7fd53736c66c29bfd600e5129f754378'
                or ab_run.sha(headline/'preflight.json')!='f41552bfd5d4780cd4a64db1024dc6e81a8c67beb11130fd7a4eaff7fdbb3d95'
                or ab_run.sha(sms/'train.zip')!=sms_entry['files']['train.zip'] or verified_directory() is None
                or sms_entry['files']['train.zip']!=rec.read(sms.parent/'preflight.json')['sms']['files']['train.zip']):
            raise ValueError('Frozen public SMS training package or verified weights changed')
        original = load_development_task(sms/'train.zip')
        menu_facts = {}
        headline_old = headline/'45102_original'
        for old_case,development in ((sms,original),(headline_old,load_development_task(headline_old/'train.zip'))):
            contract = TaskSpec.from_task(development).record()
            caps = ab_run.registered_capabilities(contract,len(development.y_train),'full')
            old_caps = rec.read(old_case/'run/run_manifest.json')['capabilities']
            if list(caps)!=[*old_caps,TEXT_KERNEL] or any(caps[k]!=v for k,v in old_caps.items()):
                raise ValueError('Menu must preserve each existing capability and append one new choice')
            new_cap = caps[TEXT_KERNEL]
            if new_cap['resource']!='local_frozen_encoder_cpu' or new_cap['classifier_fits']!=1 or new_cap['refit_classifier_fits']!=1:
                raise ValueError('New method has the wrong execution budget or resource type')
            with patch.dict(os.environ,SUPERAGENT_MINILM_WEIGHTS=''):
                if TEXT_KERNEL in ab_run.registered_capabilities(contract,len(development.y_train),'full'):
                    raise ValueError('A locally unavailable method entered the executable menu')
            menu_facts[development.kind] = dict(old=len(old_caps),new=len(caps),capability=new_cap)
            if development.kind=='pair_text':
                symmetric = deepcopy(contract)
                symmetric['input_schema'].update(pair_semantics='symmetric',pair_label_one='equivalent')
                if TEXT_KERNEL in ab_run.registered_capabilities(symmetric,len(development.y_train),'full'):
                    raise ValueError('A directional method entered the symmetric menu')
        text = np.asarray(original.X_train)
        labels = np.asarray(original.y_train)
        rng = np.random.default_rng(88017)
        left,right = [rng.permutation(np.flatnonzero(labels==value))[:72] for value in np.unique(labels)]
        single = np.r_[left[:56],right[:56]]
        pairs = [(text[a],text[b]) if i%2 else (text[b],text[a]) for i,(a,b) in enumerate(zip(left,right,strict=True))]
        cases = {
            'single_text':(text[single].tolist(),labels[single],text[np.r_[left[56:],right[56:]]].tolist(),MetricSpec.for_task(original)),
            'pair_text':(pairs[:56],np.arange(56)%2,pairs[56:],MetricSpec('accuracy',(0,1)))
        }
        reports = {}
        for kind,(training,y,queries,metric) in cases.items():
            case = out/kind
            case.mkdir()
            payload = case/'payload.joblib'
            artifact = case/'model.joblib'
            joblib.dump(dict(kind=kind,X_train=training,y_train=y,X_search=queries[:1],mode='refit',metric_spec=metric.record(),
                             pair_semantics='preference' if kind=='pair_text' else None),payload)
            joblib.dump(queries,case/'queries.joblib')
            rec.write(case/'admission.json',dict(maximum_classifier_fits=1,ledger_path=str(case/'fit.json')))
            pilot.child(['-m','superagent.experiment_worker',payload,TEXT_KERNEL,artifact,'--fit-admission',case/'admission.json'],
                        case/'worker.log',300)
            ledger = rec.read(case/'fit.json')
            model = joblib.load(artifact)
            if ledger['completed_classifier_fits']!=1 or model['metrics']['classifier_fits']!=1:
                raise ValueError('Expected one real supervised fit')
            moved = case/'moved/model.joblib'
            moved.parent.mkdir()
            shutil.copy2(artifact,moved)
            for i,path in enumerate((artifact,moved)):
                pilot.child(['-m','scripts.check_b_text_kernel','reload','--kind',kind,'--queries',case/'queries.joblib',
                            '--model',path,'--output',case/f'pred_{i}.json'],case/f'predict_{i}.log',150)
            outcome = rec.read(case/'pred_0.json')
            if outcome!=rec.read(case/'pred_1.json') or ab_run.sha(moved)!=ab_run.sha(artifact):
                raise ValueError('Relocated classifier probabilities changed')
            features = model['model'].named_steps['features']
            reports[kind] = dict(optimization_rows=len(y),model_sha256=ab_run.sha(artifact),worker=ledger,delivery=outcome,
                                 median_squared_distance=features.median_squared_distance_,gamma=features.kernel_.gamma,usage=features.usage_)
        old = headline/'45102_original'
        source = rec.read(headline/'preflight.json')['entries'][old.name]
        if ab_run.sha(headline/'45102/features.zip')!=source['features_sha256']:
            raise ValueError('Original feature packet changed')
        legacy = out/'previous_predictions/rows.csv'
        pilot.child(['-m','superagent','predict',headline/'45102/features.zip','--run-dir',old/'relocated','--output',legacy],
                    out/'old_cli.log',300)
        csv = ab_run.sha(legacy)
        if csv!=rec.read(old/'delivery.json')['csv_sha256'] or csv!=ab_run.sha(old/'export_0/rows.csv'):
            raise ValueError('Existing model changed predictions after feature-map extension')
        if [row['candidate'] for row in rec.trace(out)]!=[TEXT_KERNEL,TEXT_KERNEL]:
            raise ValueError('Physical fit trace differs from the two admitted workers')
        rec.write(out/'results.json',dict(workers=reports,menus=menu_facts,prior_headline_rows=source['rows'],prior_headline_predictions_sha256=csv))
        rec.write(out/'completion.json',dict(status='completed',new_supervised_fits=2,fresh_new_model_reloads=4,
                  previous_model_cli_exports=1,new_grades=0,new_model_api_requests=0,new_course_rmb=0))
    except Exception as exc:
        rec.write(out/'completion.json',dict(status='failed',reason=str(exc),physical_fit_starts=len(rec.trace(out))))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest='mode',required=True)
    study = modes.add_parser('study')
    for name in ('weights','sms','headlines','output'):
        study.add_argument('--'+name,type=Path,required=True)
    reader = modes.add_parser('reload')
    reader.add_argument('--kind',required=True)
    for name in ('model','queries','output'):
        reader.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    if args.mode=='study':
        check(args)
    else:
        reload_model(args)
