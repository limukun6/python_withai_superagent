"""Compare the original A27 release and the jointly integrated release on sealed MRPC training rows."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def training_spec(packet):
    from superagent import ab_run
    from superagent.contracts import TaskSpec
    from superagent.data import load_development_task
    task=load_development_task(packet)
    contract=TaskSpec.from_task(task).record()
    if task.X_val is not None or task.y_val is not None or contract['input_schema'].get('pair_semantics')!='symmetric':
        raise ValueError('Expected training-only equivalence data')
    return task,contract,ab_run.registered_capabilities(contract,len(task.y_train),'full')


def inspect_runtime(args):
    from superagent import ab_run
    from superagent.ab_sentence import verified_directory
    if Path(ab_run.__file__).resolve().parents[1]!=args.source.resolve() or verified_directory() is None:
        raise ValueError('Wrong runtime or pinned encoder files')
    _,contract,menu=training_spec(args.packets/'train.zip')
    write(args.output,dict(sources=ab_run.sources(),contract=contract,menu=menu))


def probabilities(args):
    import joblib
    import numpy as np
    from superagent import ab_run
    from superagent.data import load_prediction_features
    from superagent.models import probability_with_model
    if Path(ab_run.__file__).resolve().parents[1]!=args.source.resolve():
        raise ValueError('Wrong probability runtime')
    task,_,_=training_spec(args.packets/'train.zip')
    selection=read(args.model/'final_selection.json')
    if sha(args.model/selection['model_file'])!=selection['model_sha256']:
        raise ValueError('Model changed before swap verification')
    model=joblib.load(args.model/selection['model_file'])['model']
    pairs=load_prediction_features(args.packets/'features.zip',task.kind,task.spec)
    reverse=load_prediction_features(args.packets/'reversed.zip',task.kind,task.spec)
    if list(reverse)!=[(b,a) for a,b in pairs]:
        raise ValueError('Blind swap does not correspond to canonical row order')
    classes,p=probability_with_model(task.kind,model,pairs)
    reverse_classes,q=probability_with_model(task.kind,model,reverse)
    if (not np.array_equal(classes,reverse_classes) or not np.isfinite(p).all() or not np.array_equal(p,q)
            or not np.allclose(p.sum(axis=1),1) or p.shape!=(len(pairs),len(classes))):
        raise ValueError('Model probabilities changed under swap or were ill-formed')
    write(args.output,dict(rows=len(pairs),classes=classes.tolist(),max_swap_probability_difference=float(np.abs(p-q).max())))


def run(args):
    import joblib
    import numpy as np
    from scripts import run_b1_ab_acceptance as rec
    from scripts.run_b_c1_matrix import read_export
    from superagent.metrics import MetricSpec
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    rec.offline(out)
    driver=Path(__file__).resolve()
    plan=driver.parents[1]/'docs/batches/BA27_01_symmetric_menu_plan.md'
    ref=read(rec.ROOT/'research/ba12_equivalence/protocol.json')['mrpc']
    packets=args.packets.resolve()
    if (any(sha(packets/f)!=v for f,v in ref['files'].items()) or
            rec.sha(gzip.decompress((packets/'row_ids.json.gz').read_bytes()))!=ref['row_ids_sha256'] or
            ref['inner_input_rows']!=2922 or ref['outer_rows']!=746):
        raise ValueError('Previously frozen, train-derived MRPC rows changed')
    sources={'a_native':args.native.resolve(),'ab_joint':rec.ROOT}
    commits={k:subprocess.check_output(['git','rev-parse','HEAD'],cwd=s,text=True).strip() for k,s in sources.items()}
    if commits['a_native']!='202a250fb234248f301ce3fed02e47f3befd9cbd':
        raise ValueError('Non-native A reference')
    env=dict(os.environ,SUPERAGENT_MINILM_WEIGHTS=str(args.weights.resolve()),SUPERAGENT_OFFLINE='1',
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    def child(key, command, log, timeout=300):
        root=sources[key]
        local=dict(env,PYTHONPATH=os.pathsep.join([str(rec.ROOT/'scripts/ab_acceptance_guard'),str(root)]))
        with log.open('x') as handle:
            p=subprocess.run([sys.executable,*map(str,command)],cwd=root,env=local,stdout=handle,stderr=subprocess.STDOUT,timeout=timeout)
        if p.returncode!=0:
            raise ValueError(key+' failed at '+log.name)
    write(out/'protocol.json',dict(batch='BA27-01',sources=commits,plan_sha256=sha(plan),runner_sha256=sha(driver),
        source_registration_sha256=sha(rec.ROOT/'research/ba12_equivalence/protocol.json'),input_sha256=ref['files'],
        inner_seeds=[707,808,909],outer_seed=45101,training=2922,outer=746,wall_seconds=1200,worker_seconds=300,
        max_fits_per_path=11,batch_fit_cap=22,model_api_cap=0))
    start=time.monotonic()
    jobs={}
    try:
        contexts={}
        for key,root in sources.items():
            folder=out/key
            folder.mkdir()
            child(key,[driver,'inspect','--source',root,'--packets',packets,'--output',folder/'context.json'],folder/'inspect.log')
            contexts[key]=read(folder/'context.json')
        native,joint=contexts.values()
        a_ids=[f'symmetric_{r}_C{c}' for r in ('word','word_char') for c in (.3,3.)]
        b_ids=['unordered_word','unordered_char','unordered_word_char','unordered_minilm_lexical']
        if (list(native['menu'])!=a_ids or list(joint['menu'])!=a_ids+b_ids or
                native['contract']!=joint['contract'] or any(native['menu'][k]!=joint['menu'][k] for k in a_ids) or
                any(entry['classifier_fits']!=1 or entry['refit_classifier_fits']!=1 or entry['swap_label_rule']!='preserves'
                    for entry in joint['menu'].values())):
            raise ValueError('Task contract, common four methods, or contributor extensions changed')
        write(out/'preflight.json',{k:dict(production_sha256=rec.digest(v['sources']),scheduler_sha256=v['sources']['ab_scheduler.py'],
                  candidates=list(v['menu']),catalog_sha256=rec.digest(v['menu']),contract_sha256=rec.digest(v['contract'])) for k,v in contexts.items()})
        for key in sources:
            folder=out/key
            run_dir=folder/'run'
            before=len(rec.trace(out))
            child(key,['-m','superagent','run',packets/'train.zip','--run-dir',run_dir,'--scheduler','ab-v1','--portfolio','ab-v1',
                '--catalog-scope','full','--history-policy','strict','--planner','local','--fit-limit','11','--seed','707',
                '--time-limit','1200','--trial-timeout','300'],folder/'train.log',1260)
            manifest=read(run_dir/'run_manifest.json')
            ledger=read(run_dir/'classifier_fits.json')
            outcome=read(run_dir/'run_outcome.json')
            selection=read(run_dir/'final_selection.json')
            for split,row in ref['splits'].items():
                current=read(run_dir/(split+'_split.json'))
                if any(current[k]!=v for k,v in row.items()) or current['search_rows_sharing_text']!=0:
                    raise ValueError('An inner row, label, text group or registered training payload differs')
            fits=ledger['completed_classifier_fits']
            if (manifest['source_files']!=contexts[key]['sources'] or manifest['capabilities']!=contexts[key]['menu'] or
                manifest['task_archive_sha256']!=ref['files']['train.zip'] or manifest['task_contract']!=contexts[key]['contract'] or
                not outcome['requested_planner_completed'] or not outcome['full_refit_completed'] or outcome['new_api_requests']!=0 or
                not 1<=fits==ledger['attempted_classifier_fits']==len(rec.trace(out))-before<=11 or ledger['unknown_failed_attempts']!=0 or
                sha(run_dir/selection['model_file'])!=selection['model_sha256']):
                raise ValueError('Candidate execution, model identity or actual fit budget differs')
            jobs[key]=dict(selected=selection['family'],model_sha256=selection['model_sha256'],fits=fits,
                actions=read(run_dir/'actions.json'),action_counts=outcome['action_counts'],seconds=outcome['elapsed_seconds'])
            write(out/'checkpoint.json',jobs)
        write(out/'all_models_frozen.json',{k:v['model_sha256'] for k,v in jobs.items()})
        predictions={}
        for key,root in sources.items():
            folder=out/key
            run_dir=folder/'run'
            selection=read(run_dir/'final_selection.json')
            moved=folder/'relocated'
            moved.mkdir()
            for f in (selection['model_file'],'task_spec.json','final_selection.json'):
                shutil.copy2(run_dir/f,moved/f)
            digests=[]
            for kind,model,packet in [('native',run_dir,'features.zip'),('relocated',moved,'features.zip'),('swap',moved,'reversed.zip')]:
                csv=folder/kind/'rows.csv'
                child(key,['-m','superagent','predict',packets/packet,'--run-dir',model,'--output',csv],folder/(kind+'.log'))
                values=read_export(csv,ref['outer_rows'])
                digests.append(sha(csv))
            if len(set(digests))!=1:
                raise ValueError('Native, relocated and swap deliveries differ')
            predictions[key]=np.array(values,dtype=int)
            jobs[key]['csv_sha256']=digests[0]
            child(key,[driver,'probabilities','--source',root,'--packets',packets,'--model',moved,'--output',folder/'swap.json'],
                  folder/'probability.log')
            jobs[key]['swap']=read(folder/'swap.json')
        if sha(packets/'holdout.joblib')!=ref['holdout_sha256']:
            raise ValueError('Frozen evaluation truth changed')
        held=joblib.load(packets/'holdout.joblib')
        features,truth=held['X'],held['y']
        from superagent.data import load_development_task, load_prediction_features
        task=load_development_task(packets/'train.zip')
        if list(features)!=list(load_prediction_features(packets/'features.zip',task.kind,task.spec)):
            raise ValueError('Grading and exported rows differ')
        metric=MetricSpec.for_task(task)
        correct={}
        for key,pred in predictions.items():
            correct[key]=truth==pred
            jobs[key].update(rows=len(truth),correct=int(correct[key].sum()),scores=metric.evaluate(truth,pred))
        old,new=correct.values()
        if sum(v['fits'] for v in jobs.values())!=len(rec.trace(out)):
            raise ValueError('Aggregate physical fit mismatch')
        write(out/'results.json',dict(paths=jobs,newly_correct=int((new&~old).sum()),newly_wrong=int((old&~new).sum()),
              delta_accuracy_pp=100*float(new.mean()-old.mean())))
        write(out/'completion.json',dict(status='completed',actual_fits=len(rec.trace(out)),fresh_cli_exports=6,swap_probability_processes=2,
            elapsed_seconds=time.monotonic()-start,model_api_requests=0,added_course_rmb=0,H_open_or_fitted_or_scored=False))
    except Exception as exc:
        write(out/'completion.json',dict(status='failed',error=str(exc),observed_fits=len(rec.trace(out)),model_api_requests=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    stages=parser.add_subparsers(dest='stage',required=True)
    for key,fields in [('study',('native','packets','weights','output')),('inspect',('source','packets','output')),
                       ('probabilities',('source','packets','model','output'))]:
        sub=stages.add_parser(key)
        for field in fields:
            sub.add_argument('--'+field,type=Path,required=True)
    opts=parser.parse_args()
    {'study':run,'inspect':inspect_runtime,'probabilities':probabilities}[opts.stage](opts)
