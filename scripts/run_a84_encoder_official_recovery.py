"""Recover only A81 official execution after correcting fixture modality, without repeating four fits."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a29_choice_matrix import launch,physical
from superagent.ab_run import sources
from superagent.data import load_training_task
from tools.export_model_bundle import export

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();frozen=sources()
    save(root/'protocol.json',dict(batch='A84',runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A84_encoder_official_recovery_plan.md'),original_protocol=read(a.reference/'protocol.json'),
        source_files=frozen,maximum_classifier_fits=5,starting_A_fits=1243,real_api_requests=0))
    env={**os.environ,'SUPERAGENT_MINILM_WEIGHTS':str(a.assets/'minilm'),
        'SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS':str(a.assets/'mobilenet_v3_small-047dcff4.pth'),
        'A74_OFFLINE':'1','A74_SOURCE_ROOT':str(ROOT),'AB_A29_OBSERVE_FITS':'1',
        'AB_A29_FIT_LOG':str(root/'physical_fits.jsonl'),'AB_A29_RUN_NAME':'A84_official',
        'PYTHONPATH':os.pathsep.join([str(ROOT/'scripts/a74_offline_guard'),str(ROOT)]),'PYTHONIOENCODING':'utf-8',
        'SUPERAGENT_OFFLINE':'1','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
        'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    os.environ.update({k:env[k] for k in ('SUPERAGENT_MINILM_WEIGHTS','SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS')})
    try:
        if frozen!=read(a.reference/'protocol.json')['source_files']:raise ValueError('A81 production source changed')
        for name in ('anonymous_train.zip','anonymous_query.zip'):
            with zipfile.ZipFile(a.reference/name) as old,zipfile.ZipFile(root/name,'x') as new:
                for member in old.namelist():
                    content=old.read(member)
                    if member=='manifest.json':
                        spec=json.loads(content)
                        if spec['modality']!='single_text':raise ValueError('Unexpected original fixture modality')
                        spec['modality']='text';content=json.dumps(spec).encode()
                    new.writestr(member,content)
        task=load_training_task(root/'anonymous_train.zip')
        if len(task.y_train)!=256 or task.kind!='single_text':raise ValueError('Corrected task identity differs')
        directory=root/'run'
        code,seconds=launch([sys.executable,'-m','superagent','run',str(root/'anonymous_train.zip'),'--run-dir',str(directory),
            '--scheduler','ab-v1','--portfolio','ab-v1','--fit-limit','5','--seed','707','--time-limit','600',
            '--trial-timeout','180','--partial-delivery','checkpoint'],root/'train.log',env,660)
        if code or not (directory/'run_outcome.json').exists():raise RuntimeError('Official run failed; see preserved train.log')
        outcome=read(directory/'run_outcome.json')
        if not outcome['requested_planner_completed']:raise RuntimeError(str(outcome.get('error')))
        fits=len(physical(root))
        if fits>5 or fits!=outcome['classifier_fit_accounting']['attempted_classifier_fits']:raise ValueError('Fit accounting differs')
        menu=read(directory/'run_manifest.json')['capabilities']
        if not all(k in menu for k in ('minilm_text_with_lexical_features','minilm_text_with_nonlinear_lexical_features')):raise ValueError('Verified encoder hidden')
        prediction=root/'prediction.csv';code,_=launch([sys.executable,'-m','scripts.a38_predict_guard',str(root/'anonymous_query.zip'),
            '--run-dir',str(directory),'--output',str(prediction)],root/'predict.log',env,180)
        if code:raise RuntimeError('New-process prediction failed')
        bundle=export(directory,root/'export');fresh=root/'new_cwd';fresh.mkdir()
        def isolated(label,environment):
            with (root/(label+'.log')).open('w',encoding='utf-8') as log:
                return subprocess.run([sys.executable,'-I',str(bundle/'predict.py'),'--features',str(root/'anonymous_query.zip'),
                    '--output',str(root/(label+'.csv'))],cwd=fresh,env=environment,stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
        if isolated('isolated',env) or (root/'isolated.csv').read_bytes()!=prediction.read_bytes():raise ValueError('Source bundle differs')
        manifest=read(bundle/'bundle_manifest.json');missing_refused=None
        if manifest['external_resources']:
            missing_refused=isolated('missing_weights',{**env,'SUPERAGENT_MINILM_WEIGHTS':'','SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS':''})!=0 and not (root/'missing_weights.csv').exists()
            if not missing_refused:raise ValueError('Missing assets produced output')
        if len(physical(root))!=fits:raise ValueError('Prediction performed training')
        save(root/'results.json',dict(official_classifier_fits=fits,process_seconds=seconds,selected_candidate=read(directory/'final_selection.json')['family'],
            executed_candidates=[r['candidate'] for r in read(directory/'history.json')],catalogue=list(menu),CSV_byte_identical=True,
            external_resources=manifest['external_resources'],dependencies=manifest['dependencies'],missing_weights_refused=missing_refused,
            original_four_worker_probes_not_repeated=True,not_performance_superiority_test=True))
        save(root/'completion.json',dict(status='passed',actual_classifier_fits=fits,real_api_requests=0,A_cumulative_fits_after=1243+fits,
            elapsed_seconds=time.monotonic()-start,source_unchanged=sources()==frozen,formal_test_or_H_opened=False))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=len(physical(root)),real_api_requests=0,
            A_cumulative_fits_after=1243+len(physical(root)),elapsed_seconds=time.monotonic()-start,automatic_retry=False));raise
