"""Measure offline frozen prediction across the pinned runtime, missing weights, and a light runtime."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.verify_b_a31_bundle import cases, call, file_hash


def run(args):
    out, work = args.output.resolve(),args.workspace.resolve()
    out.mkdir(parents=True,exist_ok=False)
    receipt = json.loads(args.verification.read_text())
    if file_hash(args.bundle)!=receipt['bundle_sha256']:
        raise ValueError('Wrong previously verified bundle')
    clone = out/'repo'
    call(['git','clone','--quiet','--no-checkout','--no-local','--no-tags',args.receiver.resolve(),clone],out)
    missing = out/'empty_weights'
    missing.mkdir()
    real_weights = work/'pretrained_weights/all-MiniLM-L6-v2'
    environments = (
        ('pinned_with_weights',sys.executable,real_weights),
        ('pinned_missing_weights',sys.executable,missing),
        ('light_with_weights',str(args.light_python.absolute()),real_weights),
    )
    plan = Path(__file__).resolve().parents[1]/'docs/batches/B_RECEIVER_ENVIRONMENT_01_plan.md'
    protocol = dict(batch='B-RECEIVER-ENVIRONMENT-01',plan_sha256=file_hash(plan),driver_sha256=file_hash(Path(__file__)),
        verification_sha256=file_hash(args.verification),bundle_sha256=receipt['bundle_sha256'],
        selected=['coupons','sms_kernel','sms_recipe'],environments=[n for n,_,_ in environments],fit_limit=0,model_api_limit=0)
    def save(name,obj):
        (out/(name+'.json')).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    save('protocol',protocol)
    results = []
    try:
        for branch,name,case,features,rows,_,_,_ in cases(work):
            if name not in protocol['selected']:
                continue
            pinned = receipt['predictions'][name]
            call(['git','switch','-q','--detach',receipt['branch_tips'][branch]],clone)
            selection = json.loads((case/'relocated/final_selection.json').read_text())
            if file_hash(case/'relocated'/selection['model_file'])!=pinned['model_sha256'] or file_hash(features)!=pinned['features_sha256']:
                raise ValueError('Frozen model or unlabeled delivery input differs from clean-receiver acceptance')
            for env_name,interpreter,weights in environments:
                route = [clone/'scripts/ab_acceptance_guard',clone]
                if env_name=='light_with_weights':
                    route.append(work/'ab_portable_inputs/runs/py314_http_import_dependencies')
                env = dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,route)),AB_B1_OFFLINE='1',
                    AB_B1_FIT_LOG=str(out/'physical_fits.jsonl'),SUPERAGENT_OFFLINE='1',HF_HUB_OFFLINE='1',
                    TRANSFORMERS_OFFLINE='1',SUPERAGENT_MINILM_WEIGHTS=str(weights),TOKENIZERS_PARALLELISM='false',
                    OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
                env.pop('SUPERAGENT_MINILM_CACHE_DIR',None)
                probe = """from importlib.metadata import version, PackageNotFoundError
from importlib.util import find_spec
import json, pathlib, platform, superagent
values = {}
for module,dist in [('sklearn','scikit-learn'),('numpy','numpy'),('scipy','scipy'),('catboost','catboost'),('torch','torch'),('transformers','transformers'),('tokenizers','tokenizers')]:
    values[module] = version(dist) if find_spec(module) else None
print(json.dumps(dict(python=platform.python_version(),packages=values,source=str(pathlib.Path(superagent.__file__).resolve()))))"""
                runtime = json.loads(call([interpreter,'-c',probe],clone,env))
                if Path(runtime.pop('source'))!=clone/'superagent/__init__.py':
                    raise ValueError('Process imported another code checkout')
                target = out/(name+'_'+env_name)
                target.mkdir()
                dest = target/'rows.csv'
                command = [interpreter,'-m','superagent','predict',str(features),'--run-dir',str(case/'relocated'),'--output',str(dest)]
                result = subprocess.run(command,cwd=clone,env=env,capture_output=True,text=True,timeout=240)
                output = result.stdout+result.stderr
                (target/'process.log').write_text(output)
                n = None
                digest = None
                if dest.is_file():
                    digest = file_hash(dest)
                    with dest.open(encoding='utf-8',newline='') as f:
                        n = sum(1 for _ in csv.DictReader(f))
                identical = digest==pinned['csv_sha256'] and n==rows
                failure = None
                if result.returncode:
                    if 'Set SUPERAGENT_MINILM_WEIGHTS' in output:
                        failure = 'missing_verified_minilm_weights'
                    elif 'ModuleNotFoundError:' in output:
                        failure = next(l.split('ModuleNotFoundError:',1)[1].strip() for l in reversed(output.splitlines()) if 'ModuleNotFoundError:' in l)
                    else:
                        failure = next((l.strip() for l in reversed(output.splitlines()) if l.strip()),'unknown').replace(str(work),'<workspace>')[:300]
                results.append(dict(case=name,environment=env_name,returncode=result.returncode,runtime=runtime,
                    rows=n,csv_sha256=digest,matches_pinned=identical,partial_output_on_failure=bool(result.returncode and dest.exists()),
                    version_warning='InconsistentVersionWarning' in output,reason=failure))
                save('checkpoint',results)
                if env_name=='pinned_with_weights' and (result.returncode!=0 or not identical):
                    raise ValueError('Pinned environment failed to reproduce the baseline')
        if len(results)!=9 or (out/'physical_fits.jsonl').exists():
            raise ValueError('Incomplete matrix or unexpected fitting')
        save('results',results)
        save('completion',dict(status='completed',processes=len(results),matching_outputs=sum(r['matches_pinned'] for r in results),
             failed_processes=sum(bool(r['returncode']) for r in results),partial_outputs_on_failure=sum(r['partial_output_on_failure'] for r in results),
             new_fits=0,new_model_api_requests=0,new_course_rmb=0,labels_read=False))
    except Exception as error:
        save('completion',dict(status='failed',reason=str(error)))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','workspace','verification','receiver','light-python','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
