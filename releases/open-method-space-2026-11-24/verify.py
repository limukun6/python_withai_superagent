"""Verify the standalone runtime, public score counts, native fit receipts and autonomous generation."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parent
E=ROOT/'evidence'
BASE='6ecaaa802c18f0f4fcbab50e6113edd71c871e08'


def read(file):
    return json.loads(file.read_text())


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def check(ok,message):
    if not ok:
        raise ValueError(message)


def hashes(root):
    return {p.relative_to(root).as_posix():sha(p) for p in root.rglob('*.py')}


def inspect():
    actual={p.relative_to(ROOT).as_posix():sha(p) for p in ROOT.rglob('*') if p.is_file() and p.name!='SHA256SUMS' and '__pycache__' not in p.parts}
    listed={n:h for h,n in (s.split('  ',1) for s in (ROOT/'SHA256SUMS').read_text().splitlines())}
    check(actual==listed,'Different release files')
    public=read(ROOT/'choices.json')
    counts=read(E/'scoring_counts.json')
    fits=actions=rankings=0
    for scope in ('matrix','contracts','released'):
        folder=E/scope
        completion=read(folder/'completion.json')
        check(all(sha(folder/(n+'.json'))==v for n,v in completion.items()),'Different completion receipts')
        protocol=read(folder/'protocol.json')
        frozen=read(folder/'models_frozen.json')
        scores=read(folder/'results.json')
        for receipt in sorted(folder.glob('*/receipt.json')):
            path=receipt.parent
            key=path.name
            slot,mode=key.split('_')
            ref=key if scope=='matrix' else slot
            check(sha(receipt)==frozen[ref],'Model freeze changed')
            record=read(receipt)
            physical=[json.loads(line) for line in (path/'physical.jsonl').read_text().splitlines()]
            check(record['api_calls']==0 and record['actual_fits']==len(physical) and physical==record['physical']
                  and 1<=len(physical)<=11 and physical[-1]['candidate']==record['winner'],'Independent supervised charges')
            cap=4 if scope=='contracts' else 11 if scope=='released' else 5 if slot=='q0' else 6
            check(len(physical)<=cap and record['actions'][-1]['action']=='stop' and
                  record['actions_replayed']==len(record['trials'])+1==len(record['actions']),'Native budget/stop')
            sets=public[key]
            check(record['winner_generated']==(record['winner'] not in sets['base']) and
                  all(t['generated']==(t['candidate'] not in sets['base']) for t in record['trials']),'Runtime origin')
            ex=(scores[slot]['predictions'] if scope=='contracts' else read(folder/'exports.json')[ref])
            check(ex[0]==ex[1]==sha(path/'export_0.csv')==sha(path/'export_1.csv'),'Independent export pair')
            with (path/'export_0.csv').open() as f:
                rows=list(csv.DictReader(f))
            check([r['row_id'] for r in rows]==list(map(str,range(len(rows)))),'Row order')
            if scope in ('matrix','released'):
                cm=np.asarray(counts[key]['counts'])
                correct=int(cm.trace())
                check(int(cm.sum())==len(rows) and
                      cm.sum(axis=0).tolist()==[sum(str(r['prediction'])==str(v) for r in rows) for v in counts[key]['classes']],
                      'Confusion counts and exported predictions disagree')
                if scope=='matrix':
                    f1=float((2*cm.diagonal()/(cm.sum(axis=0)+cm.sum(axis=1))).mean())
                    check(scores[key]['correct']==correct and abs(scores[key]['macro_f1']-f1)<1e-12 and
                          scores[key]['primary_metric']==protocol['tasks'][slot]['metric'],'Primary metric changed')
                else:
                    cells=np.asarray(counts[key]['paired'])
                    check(scores[slot]['open_correct']==correct==cells[:,1].sum() and scores[slot]['old_correct']==cells[1,:].sum()
                          and (scores[slot]['gained'],scores[slot]['lost'])==(cells[0,1],cells[1,0]),'Released pair scoring')
            if scope=='contracts' and slot in ('t3','t4'):
                labels=[r['prediction'] for r in rows]
                check(sum((a==b) if slot=='t3' else (a!=b) for a,b in zip(labels[::2],labels[1::2]))==12,'Pair semantics')
            if scope=='contracts' and slot=='t1':
                mat=np.asarray([[float(r[f'score_{i}']) for i in range(3)] for r in rows])
                check(mat.shape==(24,3) and np.isfinite(mat).all() and np.all((mat>=0)&(mat<=1)) and np.allclose(mat.sum(axis=1),1.),'Class probabilities')
            fits+=len(physical)
            actions+=record['actions_replayed']
            rankings+=sum(s['action']=='explore' for s in sets['steps'])
    follow=ROOT/'followup'
    alloc=read(follow/'allocation/results.json')
    check(sha(follow/'allocation/paths.json')==alloc['paths_sha256'] and not alloc['advance_to_training'] and
          alloc['api_calls']==alloc['supervised_fits']==0 and len(alloc['tasks'])==5,'Allocation result')
    routes=read(follow/'allocation/paths.json')
    check(routes['plan']==sha(follow/'allocation/protocol.md') and routes['runner']==sha(follow/'allocation/replay_b_open_allocation.py'),
          'Allocation source receipt')
    shared=follow/'shared-menu'
    report=read(shared/'result.json')
    meta=read(shared/'protocol.json')
    menus=read(shared/'menu_receipts.json')
    meter=[json.loads(s) for s in (shared/'physical.jsonl').read_text().splitlines()]
    check(sha(shared/'protocol.json')==report['protocol'] and sha(shared/'menu_receipts.json')==report['menu_receipts'] and
          meta['plan']==sha(shared/'protocol.md') and meta['runner']==sha(shared/'check_b_open_shared.py') and
          len(meter)==report['actual_fits']==report['native_actions']==4 and report['real_api_calls']==0 and
          all(v['visible']['raw']==v['visible']['checked'] and v['full_equals_decoded'] and v['generated_visible']>0 and
              v['request_cap_local'] and v['live_flash_calls']==0 for v in menus.values()),'Shared planner-engineering evidence')
    check(report['exports']==[sha(shared/f'export_{i}.csv') for i in (0,1)] and
          report['exports'][0]==report['exports'][1]==sha(E/'matrix/q0_open/export_0.csv'),'Shared fixture exports')
    ext=follow/'installed-providers'
    addition=read(ext/'result.json')
    setup=read(ext/'protocol.json')
    contracts=read(ext/'contracts.json')
    trials=read(ext/'actions.json')
    native=[json.loads(s) for s in (ext/'physical.jsonl').read_text().splitlines()]
    explicit=[json.loads(s) for s in (ext/'interface_physical.jsonl').read_text().splitlines()]
    old=read(ext/'first_attempt.json')
    check(sha(ext/'protocol.json')==addition['protocol'] and sha(ext/'contracts.json')==addition['contracts'] and
          setup['runner']==sha(ext/'check_b_installed_methods.py') and setup['fixture']==sha(ext/'provider_fixture.py') and
          setup['plan']==sha(ext/'protocol.md') and old['classifier_fit_accounting']['attempted_classifier_fits']==0 and
          old['new_api_requests']==addition['new_model_api_calls']==0,'Installed-provider protocol/first attempt')
    check(len(native)==addition['native_fit_count']==4 and addition['native_fits_on_installed']==0 and
          len(explicit)==addition['explicit_interface_fits']==1 and explicit[0]['candidate']==addition['explicit_interface_candidate'] and
          native[-1]['candidate']==addition['native_winner'] and trials[-1]['action']=='stop' and addition['version_mismatch_rejected'],
          'Installed-provider native/interface evidence')
    explored=[a['candidate_id'] for a in trials if a['action']=='explore']
    check(explored==[v['chosen'] for v in addition['choices']]==[v['candidate'] for v in native[:-1]] and
          all(len(v['installed_eligible'])==2 and v['chosen'] not in v['installed_eligible'] for v in addition['choices']),
          'Native planner eligibility and decisions')
    rules=contracts['contracts']
    check(len(contracts['installed'])==2 and len(contracts['invalid'])==1 and len(rules['binary']['visible'])==2 and
          rules['multiclass']['visible']==rules['probabilities']['visible']==['MedianPrototypeClassifier'],
          'Installed-provider filter contract')
    for prefix in ('native','interface'):
        exports=[ext/f'{prefix}_{i}.csv' for i in (0,1)]
        check(list(map(sha,exports))==addition[prefix+'_exports'] and sha(exports[0])==sha(exports[1]) and
              [r['row_id'] for r in csv.DictReader(exports[0].read_text().splitlines())]==list(map(str,range(33))),
              'Installed-provider export pair')
    return fits+len(meter)+len(native)+len(explicit),actions+report['native_actions']+len(trials),rankings+len(explored)


def check_runtime(repo):
    with TemporaryDirectory() as temp:
        runtime=Path(temp)/'runtime'
        with zipfile.ZipFile(ROOT/'runtime-source.zip') as z:
            z.extractall(runtime)
        p=read(E/'matrix/protocol.json')
        check(hashes(runtime/'superagent')==p['source'] and sha(runtime/'scripts/ab_acceptance_guard/sitecustomize.py')==p['guard'],'Standalone source')
        for scope,script in [('matrix','run_b_open_methods'),('contracts','check_b_open_contracts'),('released','run_b_open_released')]:
            meta=read(E/scope/'protocol.json')
            check(meta['source']==p['source'] and sha(runtime/'scripts'/(script+'.py'))==meta['runner'],'Exact measured runtime and experiment runner')
        subprocess.run([sys.executable,'-m','scripts.replay_b_open_choices',str(ROOT/'choices.json')],cwd=runtime,check=True)
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests'],cwd=runtime,check=True)
        ext=ROOT/'followup/installed-providers'
        latest=Path(temp)/'extensible'
        with zipfile.ZipFile(ext/'runtime-source.zip') as z:
            z.extractall(latest)
        expected=read(ext/'protocol.json')
        check(hashes(latest/'superagent')==expected['source'] and
              sha(latest/'scripts/check_b_installed_methods.py')==expected['runner'] and
              sha(latest/'scripts/provider_fixture.py')==expected['fixture'],'Extensible standalone runtime')
        shutil.copy(runtime/'scripts/replay_b_open_choices.py',latest/'scripts')
        replay=[sys.executable,'-m','scripts.replay_b_open_choices']
        subprocess.run(replay+[str(ROOT/'choices.json')],cwd=latest,check=True)
        install=Path(temp)/'extensions'
        subprocess.run([sys.executable,'-c',
                        "from pathlib import Path; from scripts.check_b_installed_methods import install_example; import sys; install_example(Path(sys.argv[1]),'1.0')",
                        str(install)],cwd=latest,check=True)
        subprocess.run(replay+[str(ext/'choices.json')],cwd=latest,env=dict(os.environ,PYTHONPATH=str(install)),check=True)
        if repo:
            base=Path(temp)/'team'
            base.mkdir()
            blob=subprocess.check_output(['git','-C',str(repo),'archive',BASE,'--','superagent','scripts/ab_acceptance_guard'])
            with tarfile.open(fileobj=io.BytesIO(blob)) as a:
                a.extractall(base,filter='data')
            subprocess.run(['git','apply',str(ROOT/'team-integration.patch')],cwd=base,check=True)
            check(hashes(base/'superagent')==p['source'] and sha(base/'scripts/ab_acceptance_guard/sitecustomize.py')==p['guard'],
                  'Integration patch does not reproduce measured code')
            shared=ROOT/'followup/shared-menu'
            current=read(shared/'protocol.json')
            staged=Path(temp)/'current'
            staged.mkdir()
            newer=subprocess.check_output(['git','-C',str(repo),'archive',current['upstream'],'--','superagent','scripts/ab_acceptance_guard'])
            with tarfile.open(fileobj=io.BytesIO(newer)) as a:
                a.extractall(staged,filter='data')
            subprocess.run(['git','apply',str(shared/'current-team.patch')],cwd=staged,check=True)
            check(hashes(staged/'superagent')==current['source'],'Current shared adapter differs from the tested local integration')
            shutil.copy(runtime/'scripts/replay_b_open_choices.py',staged/'scripts')
            subprocess.run([sys.executable,'-m','scripts.replay_b_open_choices',str(ROOT/'choices.json')],cwd=staged,check=True)
            provider_base=Path(temp)/'provider-patch'
            provider_base.mkdir()
            with tarfile.open(fileobj=io.BytesIO(newer)) as a:
                a.extractall(provider_base,filter='data')
            subprocess.run(['git','apply',str(ext/'current-team.patch')],cwd=provider_base,check=True)
            check(hashes(provider_base/'superagent')==expected['source'] and
                  sha(provider_base/'scripts/ab_acceptance_guard/sitecustomize.py')==sha(latest/'scripts/ab_acceptance_guard/sitecustomize.py') and
                  sha(provider_base/'scripts/check_b_installed_methods.py')==expected['runner'] and
                  sha(provider_base/'scripts/provider_fixture.py')==expected['fixture'] and
                  sha(provider_base/'docs/batches/B_runtime_provider_20261124_zh.md')==expected['plan'],
                  'Latest team patch differs from the tested extensible runtime')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path)
    args=parser.parse_args()
    totals=inspect()
    check_runtime(args.repo.resolve() if args.repo else None)
    print('Verified %d supervised fits, %d native action receipts and %d exploration rankings'%totals)
