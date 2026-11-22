"""Verify offline visual resource handling against the actual frozen photo run."""
import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

from scripts import run_b1_ab_acceptance as fixture
from superagent.ab_catalog import capabilities_for
from superagent.ab_run import sources
from superagent.ab_vision import WEIGHT_ENV, WEIGHT_SHA256
from superagent.contracts import TaskSpec
from superagent.data import load_development_task


def check(root, source, weights):
    root.mkdir(parents=True,exist_ok=False)
    fixture.offline(root)
    ref = fixture.ROOT/'research/bv_01_photos'
    expected = fixture.read(ref/'results.json')['paths']
    registry = fixture.read(ref/'protocol.json')['tasks']
    plan = [('vision','verified'),('vision','missing'),('vision','changed'),('existing','missing')]
    fixture.write(root/'protocol.json',dict(batch='BV-02', cases=plan,models={k:v['model_sha256'] for k,v in expected.items()},
        plan_sha256=fixture.sha((fixture.ROOT/'docs/batches/BV_02_delivery_plan.md').read_bytes()),
        driver_sha256=fixture.sha(Path(__file__).read_bytes()),source_sha256=fixture.digest(sources()),fits_allowed=0,api_allowed=0))
    results = {}
    try:
        installed = root/'copied_weights.pth'
        shutil.copyfile(weights, installed)
        if fixture.sha(installed.read_bytes()) != WEIGHT_SHA256:
            raise ValueError('Different upstream source weights')
        changed = root/'changed_weights.pth'
        altered = bytearray(installed.read_bytes())
        altered[-1] ^= 1
        changed.write_bytes(altered)
        exports = {'verified':installed,'changed':changed,'missing':None}
        for arm,case in plan:
            run = source/arm/'run'
            feature_file = source/arm/'features.zip'
            working = root/f'{arm}_{case}'
            working.mkdir()
            chosen = fixture.read(run/'final_selection.json')
            for name in ('task_spec.json','final_selection.json',chosen['model_file']):
                shutil.copyfile(run/name,working/name)
            if (fixture.sha((working/chosen['model_file']).read_bytes()) != expected[arm]['model_sha256'] or
                    fixture.sha(feature_file.read_bytes()) != registry[arm]['files']['features.zip']):
                raise ValueError('Frozen model or inputs changed')
            with patch.dict(os.environ,{WEIGHT_ENV:str(installed) if arm=='vision' else ''}):
                training = source/arm/'train.zip'
                if fixture.sha(training.read_bytes()) != registry[arm]['files']['train.zip']:
                    raise ValueError('Training input differs from the original manifest')
                task = load_development_task(training)
                caps = capabilities_for(TaskSpec.from_task(task).record(),train_rows=len(task.y_train))
            if fixture.digest(caps) != registry[arm]['catalog_sha256'] or list(caps) != registry[arm]['candidate_ids']:
                raise ValueError('Current eligibility guard changed this actual photo catalogue')
            env = dict(os.environ, TORCH_HOME=str(working/'empty_torch_home'),OMP_NUM_THREADS='2')
            env.pop(WEIGHT_ENV,None)
            if exports[case] is not None:
                env[WEIGHT_ENV] = str(exports[case])
            out = working/'predictions.csv'
            proc = subprocess.run([sys.executable,'-m','superagent','predict',str(feature_file),'--run-dir',str(working),'--output',str(out)],
                                  env=env,cwd=working,capture_output=True,text=True,timeout=300)
            (working/'predict.log').write_text(proc.stdout+proc.stderr,encoding='utf-8')
            if arm=='vision' and case!='verified':
                if (proc.returncode==0 or out.exists() or (working/'prediction_manifest.json').exists()
                        or WEIGHT_ENV not in proc.stderr+proc.stdout):
                    raise ValueError('Unusable resource was not rejected before model export')
                result = dict(status='explicit_rejection',exit_code=proc.returncode)
            else:
                if proc.returncode or not out.is_file() or fixture.sha(out.read_bytes()) != expected[arm]['delivery']['csv_sha256']:
                    raise ValueError('Model export differs from the original frozen predictions')
                with out.open(encoding='utf-8',newline='') as stream:
                    ids = [r['row_id'] for r in csv.DictReader(stream)]
                if ids != list(map(str,range(expected[arm]['delivery']['rows']))):
                    raise ValueError('Unexpected row count or ordering')
                result = dict(status='identical_export',rows=len(ids),csv_sha256=fixture.sha(out.read_bytes()))
            results[working.name] = dict(result, training_catalogue_matches=True)
            fixture.write(root/'results.json',results)
        fixture.write(root/'completion.json',dict(status='passed',processes=len(results),exact_exports=2,expected_rejections=2,
                      fits=0,outer_scores_read=0,model_api_requests=0))
    except Exception as error:
        fixture.write(root/'completion.json',dict(status='failed',reason=str(error),completed_cases=list(results),fits=0,model_api_requests=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('output','source','weights'):
        parser.add_argument('--'+key,type=Path,required=True)
    a=parser.parse_args()
    check(a.output.resolve(),a.source.resolve(),a.weights.resolve())
