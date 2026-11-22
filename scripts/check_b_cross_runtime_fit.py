"""Run the two-fit portable-input qualification using two actual Python installations."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess

from scripts import run_b1_ab_acceptance as fixture
from superagent.ab_run import sources
from superagent.prediction_evidence import fingerprint


REF = fixture.ROOT/'research/bp_02_materialization'


def execute(command, env, log, timeout):
    with log.open('x',encoding='utf-8') as stream:
        p=subprocess.run(command, env=env, cwd=fixture.ROOT, stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
    if p.returncode:
        raise RuntimeError(log.name+' returned '+str(p.returncode))


def verified_inputs(which, supplied):
    records=fixture.read(REF/which/'results.json')['bean']
    physical={'train.zip':supplied/'bean/optimizer/train.zip','inner_splits.json':supplied/'bean/optimizer/inner_splits.json',
              'features.zip':supplied/'bean/delivery/features.zip'}
    for file,path in physical.items():
        if fixture.sha(path.read_bytes()) != records['canonical_files'][file]:
            raise ValueError('Portable source differs from registered input in '+which+'/'+file)
    return records,physical


def check(args):
    root = args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    fixture.offline(root)
    frozen=fixture.digest(sources())
    runs = {}
    environments = {}
    try:
        for i,which in enumerate(('environment_1','environment_2'),1):
            work=root/which
            work.mkdir()
            records,files=verified_inputs(which,getattr(args,'inputs_'+str(i)).resolve())
            # Keep venv interpreter paths intact so Python activates the corresponding installation.
            python = str(getattr(args,'python_'+str(i)).absolute())
            paths = [str(fixture.ROOT/'scripts/ab_acceptance_guard'),str(fixture.ROOT)]
            if i==2 and args.overlay:
                paths.append(str(args.overlay.resolve()))
            env=dict(os.environ,PYTHONPATH=os.pathsep.join(paths),AB_B1_FIT_LOG=str(work/'physical_fits.jsonl'),
                     OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            execute([python,'-c',"import json,platform,requests; from importlib.metadata import version; print(json.dumps({'python':platform.python_version(), **{k:version(k) for k in ('numpy','scipy','scikit-learn','joblib')}}))"],
                    env,work/'preflight.json',30)
            installed=fixture.read(work/'preflight.json')
            expected=fixture.read(REF/which/'protocol.json')['environment']
            if installed != dict(python=expected['python'],**{k:expected['packages'][k] for k in installed if k!='python'}):
                raise ValueError('Different environment from the verified BP-02 materialization')
            runs[which]=dict(python=python,env=env,files=files,records=records)
            environments[which]=installed
        fixture.write(root/'protocol.json',dict(batch='BP-03',source_sha256=frozen,environments=environments,
            input_files={k:r['records']['canonical_files'] for k,r in runs.items()},
            driver_sha256=fixture.sha(Path(__file__).read_bytes()),adapter_sha256=fixture.sha((fixture.ROOT/'tools/ab_registered_run.py').read_bytes()),
            plan_sha256=fixture.sha((fixture.ROOT/'docs/batches/BP_03_cli_plan.md').read_bytes()),fit_limit_per_path=2,model_api_allowed=0))
        paths={}
        for which,r in runs.items():
            work=root/which
            out=work/'run'
            cmd=[r['python'],'-m','tools.ab_registered_run','--split-manifest',str(r['files']['inner_splits.json']),
                 '--split-sha256',r['records']['canonical_files']['inner_splits.json'],'--','run',str(r['files']['train.zip']),
                 '--scheduler','ab-v1','--portfolio','ab-v1','--seed','707','--fit-limit','2','--time-limit','120',
                 '--trial-timeout','60','--run-dir',str(out)]
            execute(cmd,r['env'],work/'train.log',150)
            manifest=fixture.read(out/'run_manifest.json')
            outcome=fixture.read(out/'run_outcome.json')
            actions=fixture.read(out/'actions.json')
            ledger=fixture.read(out/'classifier_fits.json')
            evidence=fixture.read(out/'prediction_evidence.json')
            if (not outcome['full_refit_completed'] or not outcome['requested_planner_completed'] or outcome['new_api_requests']!=0
                    or not outcome['source_unchanged'] or fixture.digest(manifest['source_files'])!=frozen
                    or manifest['task_archive_sha256']!=r['records']['canonical_files']['train.zip'] or manifest['fit_limit']!=2
                    or [a['action'] for a in actions]!=['explore','stop'] or len(fixture.trace(work))!=2
                    or ledger['attempted_classifier_fits']!=2 or ledger['completed_classifier_fits']!=2 or ledger['unknown_failed_attempts']
                    or len(evidence)!=1 or fingerprint(evidence[0]['predictions'])!=evidence[0]['prediction_sha256']):
                raise ValueError('Execution, training or evidence mismatch in '+which)
            splits={}
            reference=fixture.read(fixture.ROOT/'research/bc1_preparation/results.json')['tasks']['bean']['splits']
            for split,expected in reference.items():
                fresh=fixture.read(out/(split+'_split.json'))
                sha=fixture.sha((out/(split+'.joblib')).read_bytes())
                if (any(fresh[k]!=expected[k] for k in ('row_ids_sha256','label_sha256','seed')) or
                        sha!=fresh['evaluation_payload_sha256'] or sha!=r['records']['splits'][split]['local_worker_sha256']):
                    raise ValueError('The official worker rows or its local bytes differ')
                splits[split]={k:fresh[k] for k in ('row_ids_sha256','label_sha256','evaluation_payload_sha256')}
            if (evidence[0]['label_sha256']!=splits['primary']['label_sha256'] or
                    evidence[0]['evaluation_payload_sha256']!=splits['primary']['evaluation_payload_sha256'] or
                    fingerprint(evidence[0]['row_ids'])!=splits['primary']['row_ids_sha256']):
                raise ValueError('Worker prediction evidence no longer refers to the registered primary split')
            selection=fixture.read(out/'final_selection.json')
            if fixture.sha((out/selection['model_file']).read_bytes())!=selection['model_sha256']:
                raise ValueError('Final refit model hash differs')
            paths[which]=dict(model_sha256=selection['model_sha256'],selected=selection['family'],splits=splits,
                actions=actions, menu_sha256=fixture.digest(manifest['capabilities']), fits=2,
                search_predictions_sha256=evidence[0]['prediction_sha256'],search_rows=len(evidence[0]['predictions']))
        fixture.write(root/'all_models_frozen.json',{k:v['model_sha256'] for k,v in paths.items()})
        exports=[]
        for which,r in runs.items():
            work=root/which
            dest=work/'predict'/'rows.csv'
            execute([r['python'],'-m','superagent','predict',str(r['files']['features.zip']),'--run-dir',str(work/'run'),'--output',str(dest)],
                    r['env'],work/'predict.log',90)
            with dest.open(newline='',encoding='utf-8') as stream:
                rows=list(csv.DictReader(stream))
            meta=fixture.read(dest.parent/'prediction_manifest.json')
            if (meta['test_labels_read'] or len(rows)!=r['records']['held_rows'] or
                    [row['row_id'] for row in rows]!=list(map(str,range(len(rows))))):
                raise ValueError('Incorrect unlabeled export identity')
            exports.append([row['prediction'] for row in rows])
            paths[which].update(export_rows=len(rows),export_sha256=fixture.sha(dest.read_bytes()))
        a,b=paths.values()
        if a['menu_sha256']!=b['menu_sha256']:
            raise ValueError('Cross-runtime capability menu differs')
        fixture.write(root/'results.json',dict(paths=paths,same_actions=a['actions']==b['actions'],same_model_bytes=a['model_sha256']==b['model_sha256'],
            same_in_search_predictions=a['search_predictions_sha256']==b['search_predictions_sha256'],
            export_rows=len(exports[0]), exported_label_disagreements=sum(x!=y for x,y in zip(*exports,strict=True)),
            same_export_bytes=a['export_sha256']==b['export_sha256']))
        fixture.write(root/'completion.json',dict(status='passed',paths=2,fits=4,model_api_requests=0,outer_labels_read=0,outer_scores=0))
    except Exception as error:
        fixture.write(root/'completion.json',dict(status='failed',reason=str(error),
            physical_fits_observed=sum(len(fixture.trace(p)) for p in root.iterdir() if p.is_dir()),model_api_requests=0))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('inputs-1','inputs-2','python-1','python-2','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--overlay',type=Path)
    check(p.parse_args())
