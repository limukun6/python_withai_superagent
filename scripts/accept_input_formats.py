"""Exercise the public solve entry and independently reload each output on synthetic, unlabeled queries."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

from scripts.unified_released_study import ROOT, read, save, sha
from scripts.verify_anonymous_planner import revision
from tests.test_format_delivery import fixtures,zipped,as_npy


def run_case(item):
    case,root,training,features,status,metric=item
    folder=root/case
    folder.mkdir()
    study_files=set(training)
    raw=zipped(folder/'p.zip',{**training,'val/labels.never':b'do not open','test/answers.never':b'do not open'})
    query=zipped(folder/'q.zip',features)
    work=folder/'run'
    physical=folder/'physical.jsonl'
    opened=folder/'opened.jsonl'
    environment=dict(os.environ,AB_B1_OFFLINE='1',SUPERAGENT_OFFLINE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                     PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts/unified_acceptance',ROOT))),
                     UNIFIED_RAW_INPUT=str(raw),UNIFIED_RAW_MEMBERS=json.dumps(sorted(study_files)),UNIFIED_RAW_LOG=str(opened),
                     AB_B1_FIT_LOG=str(physical),AB_B1_PER_WORKER_FIT_LIMIT='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2')
    for flag in ('SUPERAGENT_PLANNER_KEY_FILE','SUPERAGENT_PLANNER_LEDGER','SUPERAGENT_TASK_BUDGET_ID'):
        environment.pop(flag,None)
    def cli(command,file):
        with (folder/file).open('x') as log:
            return subprocess.run([sys.executable,'-m','unified_agent',*map(str,command)],cwd=ROOT,env=environment,
                                  stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,timeout=1250).returncode
    dest=folder/'delivered/predictions.csv'
    code=cli(['solve',raw,'--prediction-input',query,'--run-dir',work,'--output',dest],'solve.log')
    native=work/'agent'
    outcome=read(native/'run_outcome.json')
    actual=outcome.get('delivery_status',outcome['status'])
    if actual!=status or bool(code)!=(status=='unavailable') or outcome['new_api_requests']!=0:
        raise ValueError(case+' status or charging disagrees with expected outcome')
    fitted=physical.read_text().splitlines() if physical.exists() else []
    if len(fitted)!=outcome['attempted_classifier_fits'] or not set(map(json.loads,opened.read_text().splitlines()))<=study_files:
        raise ValueError(case+' fit or member accounting mismatch')
    receipt=dict(delivery_status=status,metric=metric,fits=len(fitted),api_calls=outcome['new_api_requests'],
                 source_members=sorted(set(map(json.loads,opened.read_text().splitlines()))))
    if status=='unavailable':
        if dest.exists() or (native/'final_selection.json').exists():
            raise ValueError('An unsupported contract produced an ungrounded output')
    else:
        if read(native/'task_spec.json')['metric']!=metric:
            raise ValueError(case+' changed the metric')
        second=folder/'reload/predictions.csv'
        if cli(['predict',query,'--run-dir',work,'--output',second],'predict.log') or sha(dest)!=sha(second):
            raise ValueError(case+' failed independent model replay')
        with dest.open() as stream:
            records=list(csv.DictReader(stream))
        expected=3 if case!='separate_target' else 2
        if [r['row_id'] for r in records]!=[str(i) for i in range(expected)]:
            raise ValueError(case+' row order changed')
        if status=='partial' and (read(dest.parent/'prediction_manifest.json')['delivery_status']!='partial'
                 or any(r['prediction']!=('3' if case=='separate_target' else 'yes') for r in records)):
            raise ValueError('Incorrect or undisclosed fallback')
        receipt.update(rows=expected,model=read(native/'final_selection.json')['model_sha256'],predictions=sha(dest))
    save(folder/'receipt.json',receipt)
    print(case,status,len(fitted),flush=True)
    return case,receipt


def main(root):
    root.mkdir(parents=True,exist_ok=False)
    cases={k:(tr,te,'completed',metric) for k,(tr,te,_,metric) in fixtures().items()}
    cases['missing_features']=({'README.md':'Target column is verdict.','train.csv':'x,verdict\n,yes\n,yes\n,no\n'},
                               {'test.csv':'x\na\nb\nc\n'},'partial','accuracy')
    manifest=json.dumps(dict(schema_version=1,modality='tabular',metric='accuracy',
        splits={'train':dict(format='npy',path='missing.npy',target_path='train_labels.npy'),'test':dict(format='npy',path='query.npy')}))
    cases['separate_target']=({'manifest.json':manifest,'train_labels.npy':as_npy([3,2,3,3])},
                              {'manifest.json':manifest,'query.npy':as_npy([[0],[1]])},'partial','accuracy')
    cases['unknown_target']=({'train.csv':'p,q\ncloud,red\nstone,blue\n'},{'test.csv':'p\ncloud\n'},'unavailable',None)
    identity=revision()
    save(root/'protocol.json',dict(source=identity,driver=sha(Path(__file__)),fixtures=sha(ROOT/'tests/test_format_delivery.py'),
         fit_limit_per_path=10,model_api_limit=0,
         expected={k:dict(status=v[2],metric=v[3]) for k,v in cases.items()},
         definition='CLI without stdin, training member allowlist, per-worker fit observer, blind queries and fresh reload.'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts=dict(pool.map(run_case,[(k,root,*v) for k,v in cases.items()]))
    if identity!=revision():
        raise ValueError('Source changed during format acceptance')
    save(root/'results.json',receipts)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out',type=Path)
    main(parser.parse_args().out.resolve())
