"""Inject delivery faults into copied metadata, never modify models or fit classifiers."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from scripts.a_batch_common import save,hashes
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a122_current_agent_acceptance import ROOT,BASE,environment,child


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--acceptance',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();source=hashes(ROOT)
    save(a.output/'protocol.json',dict(batch='A123-offline-delivery',source_files=source,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A120_A124_next_5h_plan_20261123_zh.md'),
        acceptance_completion_sha256=sha(a.acceptance/'completion.json'),maximum_classifier_fits=0,
        real_API_requests=0,outer_labels_opened=False,original_models_modified=False))
    try:
        from scripts import a38_predict_guard
        import numpy as np
        from superagent.predict import validate_prediction_outputs
        if read(a.acceptance/'completion.json')['status']!='completed':raise ValueError('Completed acceptance required')
        entries=read(a.acceptance/'results.json');prepared=read(BASE/'runs/a122_prepared_v2/results.json')
        queries={r['task']:r['features'] for r in prepared['entries']};env=environment(a.output);results=[]
        for entry in entries:
            name=entry['task'];folder=Path(entry.get('reused_from',str(a.acceptance/name)));run=folder/'run'
            selection=read(run/'final_selection.json');model=run/selection['model_file'];original_sha=sha(model)
            if original_sha!=selection['model_sha256']:raise ValueError('Original model changed')
            for fault,expected in [('missing_model','No such file or directory'),('schema_changed','Task schema no longer matches'),
                                   ('model_corrupt','Model file no longer matches')]:
                target=a.output/(name+'_'+fault);target.mkdir()
                shutil.copy2(run/'task_spec.json',target/'task_spec.json');shutil.copy2(run/'final_selection.json',target/'final_selection.json')
                if fault=='schema_changed':
                    spec=read(target/'task_spec.json');spec['metric']='tampered';save(target/'task_spec.json',spec)
                elif fault=='model_corrupt':(target/selection['model_file']).write_bytes(b'corrupted_model_fixture')
                destination=target/'predictions.csv'
                process=child([sys.executable,'-m','scripts.a38_predict_guard',queries[name],'--run-dir',str(target),
                    '--output',str(destination)],target/'failure.log',env,180,allow_failure=True)
                message=(target/'failure.log').read_text(encoding='utf-8')
                if not process['returncode'] or destination.exists() or expected not in message:
                    raise ValueError('Unexpected failure semantics: '+name+'/'+fault)
                results.append(dict(task=name,fault=fault,rejected_before_output=True,expected_error=expected))
            resources=entry['external_resources']
            if resources:
                target=a.output/(name+'_missing_weights');target.mkdir()
                extra={r['environment_variable']:'' for r in resources}
                process=child([sys.executable,'-I',str(folder/'portable/bundle/predict.py'),'--features',queries[name],
                    '--output',str(target/'predictions.csv')],target/'failure.log',{**env,**extra},180,allow_failure=True)
                message=(target/'failure.log').read_text(encoding='utf-8')
                if not process['returncode'] or (target/'predictions.csv').exists() or 'Traceback' not in message:
                    raise ValueError('Missing assets did not reject before output')
                results.append(dict(task=name,fault='missing_weights',rejected_before_output=True,
                    error_tail=message.splitlines()[-1]))
            if sha(model)!=original_sha:raise ValueError('Fault injection modified original model')
        spec={'metric':'roc_auc','metric_spec':{'name':'roc_auc','labels':[0,1],'positive_label':1,'direction':'maximize'}}
        faults=[('row_count',dict(predictions=np.array([0]),rows=2,specification=spec)),
                ('unknown_label',dict(predictions=np.array([9]),rows=1,specification=spec)),
                ('probability_shape',dict(predictions=np.array([0]),rows=1,specification=spec,classes=[0,1],scores=[[1.]])),
                ('probability_normalization',dict(predictions=np.array([0]),rows=1,specification=spec,classes=[0,1],scores=[[.4,.4]]))]
        for fault,kwargs in faults:
            try:validate_prediction_outputs(**kwargs)
            except ValueError as error:results.append(dict(task='controlled_output_contract',fault=fault,error=str(error),rejected=True))
            else:raise ValueError('Invalid output accepted: '+fault)
        if hashes(ROOT)!=source:raise ValueError('Source changed')
        save(a.output/'results.json',results)
        save(a.output/'completion.json',dict(status='passed',faults=len(results),actual_classifier_fits=0,real_API_requests=0,
            source_unchanged=True,outer_labels_opened=False,elapsed_seconds=time.monotonic()-start))
        print(json.dumps(dict(status='passed',faults=len(results),classifier_fits=0,API=0)),flush=True)
    except Exception as error:
        save(a.output/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,real_API_requests=0))
        raise


if __name__=='__main__':main()
