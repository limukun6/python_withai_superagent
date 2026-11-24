"""Verify fit receipts, exports and grouped scores; optionally apply the patch and replay choices."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

import numpy as np

ROOT=Path(__file__).resolve().parent
E=ROOT/'evidence'
BASE='9eb9b5a71f9ddce912550a28a17c12654dcd53a9'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(value,reason):
    if not value:
        raise ValueError(reason)


def receipt():
    files={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*')
           if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts}
    recorded={name:digest for digest,name in (v.split('  ',1) for v in (ROOT/'SHA256SUMS').read_text().splitlines())}
    check(files==recorded,'Release files differ from checksums')
    result=read(E/'results.json')
    protocol=read(E/'protocol.json')
    audit=read(E/'decision_audit.json')
    check(sha(E/'decision_inputs.json')==audit['portable_input_sha256'],'Exploration input digest')
    table=read(E/'table/results.json')
    check(all(sha(E/'table'/k)==v for k,v in table['proof'].items()),'Table evidence digest')
    sealed=read(E/'models_frozen.json')
    for name in ('smoke','reference','expanded','transfer','table/path'):
        folder=E/name
        is_table=name.startswith('table/')
        frozen=read(folder/'frozen.json')
        physical=[json.loads(line) for line in (folder/'physical.jsonl').read_text().splitlines()]
        info=read(E/'table/decision_audit.json') if is_table else audit['native'][name]
        exports=(read(E/'table/exports.json')['predictions'] if is_table else
                 read(folder/'delivery.json')['exports'] if name=='smoke' else read(E/'exports.json')['predictions'][name])
        if not is_table:
            check(sealed['arms'][name]==sha(folder/'frozen.json'),'Sealed study')
        check(frozen['physical']==physical and 0<len(physical)<=11 and len(physical)==info['fit_count']
              and physical[-1]['candidate']==info['selected']==frozen['selected'],'Actual fit or final selection')
        check(len(frozen['actions'])==info['actions']==info['stable_reordered_actions'] and frozen['actions'][-1]['action']=='stop','Action replay')
        check(exports[0]==exports[1]==sha(folder/'export_0/rows.csv')==sha(folder/'export_1/rows.csv'),'Independent predictions')
        with (folder/'export_0/rows.csv').open() as f:
            rows=list(csv.DictReader(f))
        expected=table['rows'] if is_table else 30 if name=='smoke' else protocol['movie_held'] if name=='transfer' else protocol['title_held']
        check(len(rows)==expected and [r['row_id'] for r in rows]==list(map(str,range(expected))) and
              all(r['prediction'] in ('0','1') for r in rows),'Prediction rows')
    cells=np.array(read(E/'title_groups.json'))
    total=cells.sum(axis=0)
    check(list(total[[2,1]])==[result['gained'],result['lost']] and
          total[1]+total[3]==result['correct']['reference'] and total[2]+total[3]==result['correct']['expanded'],'Paired scores')
    diff,size=cells[:,2]-cells[:,1],cells.sum(axis=1)
    rng=np.random.default_rng(51803)
    index=rng.integers(0,len(cells),size=(5000,len(cells)))
    bounds=np.quantile(100*diff[index].sum(axis=1)/size[index].sum(axis=1),[.025,.975])
    check(np.allclose(bounds,result['paired_group_95_pp']) and len(cells)==result['title_groups'] and sum(size)==result['title_rows'],'Grouped interval')
    check(result['model_api_calls']==table['model_api_calls']==0 and not result['formal_test_read'],'Usage and formal-test boundary')
    feedback=read(E/'feedback_results.json')
    check(feedback['paths_sha256']==sha(E/'feedback_paths.json') and feedback['new_supervised_fits']==0,'Feedback comparison')
    return sum(result['fits'].values())+table['actual_fits']


def source_check(repo,rev):
    with TemporaryDirectory() as scratch:
        folder=Path(scratch)
        tests=['tests/test_ab_scheduler.py','tests/test_ab_local_text_entry.py','tests/test_ab_text_recipes.py']
        archive=subprocess.check_output(['git','-C',str(repo),'archive',rev,'--','superagent','examples/ab_v1',
                                          'scripts/ab_acceptance_guard',*tests])
        with tarfile.open(fileobj=io.BytesIO(archive)) as t:
            t.extractall(folder,filter='data')
        subprocess.run(['git','apply',str(ROOT/'optional-methods.patch')],cwd=folder,check=True)
        shutil.copytree(ROOT/'overlay',folder,dirs_exist_ok=True)
        if subprocess.check_output(['git','-C',str(repo),'rev-parse',rev],text=True).strip()==BASE:
            p=read(E/'protocol.json')
            expected={f.relative_to(folder/'superagent').as_posix():sha(f) for f in (folder/'superagent').rglob('*.py')}
            check(expected==p['source'] and sha(folder/'scripts/ab_acceptance_guard/sitecustomize.py')==p['physical_guard'],'Exact trained source')
        for e in (E,E/'table'):
            subprocess.run([sys.executable,'-m','scripts.audit_b_varied_decisions','portable','--out',str(e)],cwd=folder,check=True)
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests'],cwd=folder,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path)
    parser.add_argument('--rev',default=BASE)
    args=parser.parse_args()
    total=receipt()
    if args.repo:
        source_check(args.repo.resolve(),args.rev)
    print('Verified',total,'supervised fits, paired scores, equal process exports and zero model API calls')
