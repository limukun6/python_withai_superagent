"""Package and independently replay both training scopes on development features before formal testing."""
import argparse
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT, read, require, save
from superagent.external_assets import activate_resource_directory
from superagent.first_delivery import digest
from tools.export_model_bundle import export
from tools.package_verified_delivery import LAUNCH, copy_resources

FILENAMES={'train_only':'Integrated-Train-Only-Task1-Task2.zip',
           'train_plus_validation':'Integrated-Train-Plus-Validation-Task1-Task2.zip'}


def independent_env():
    keep=('PATH','SYSTEMROOT','WINDIR','TMPDIR','TEMP','TMP')
    return dict({k:os.environ[k] for k in keep if k in os.environ},
                OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TOKENIZERS_PARALLELISM='false')


def native_replay(out,native,features,path):
    environment=independent_env()
    environment.update(PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts/unified_acceptance',ROOT))),
                       AB_B1_OFFLINE='1',SUPERAGENT_OFFLINE='1',HF_HUB_OFFLINE='1')
    program=('from superagent.external_assets import activate_resource_directory; '
             'from superagent.predict import create_predictions; import sys; '
             'activate_resource_directory(sys.argv[1]); create_predictions(*sys.argv[2:])')
    subprocess.run([sys.executable,'-c',program,str(out/'weights'),str(features),str(native),str(path)],
                   cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,check=True,timeout=600)


def package(out,text_election,arm):
    archive=out/FILENAMES[arm]
    receipt=out/(arm+'_portable.json')
    if receipt.is_file():
        require(digest(archive)==read(receipt)['archive_sha256'],'Existing package identity differs')
        return
    solver=source_identity(ROOT)
    activate_resource_directory(out/'weights')
    with tempfile.TemporaryDirectory(dir=out) as temp:
        work=Path(temp)
        tree=work/'delivery'
        verified={}
        for case,task in [('a','Task1'),('b','Task2')]:
            folder=tree/task
            folder.mkdir(parents=True)
            original=out/case if case=='a' else text_election
            evidence=read(out/('a' if case=='a' else 'registered/b')/'sealed.json')
            require(solver==evidence['solver'] and evidence['independent_reload'] and
                    not evidence['course_test_opened'],'Unverified original')
            augmented=arm=='train_plus_validation'
            if augmented:
                refit=read(out/'augmented'/case/'refit.json')
                native=out/'augmented'/case/'native'
                require(refit['base_model']==evidence['model'] and refit['solver']==solver
                        and not refit['official_test_opened'] and not refit['selection_repeated'],'Refit election mismatch')
                model=refit['new_model']
                rows=refit['total_training_rows']
                source_run=refit
                reference=work/(case+'_native.csv')
                native_replay(out,native,out/case/'queries.zip',reference)
                designated=None
            else:
                native=original/'run/agent'
                model=evidence['model']
                rows=evidence['training_rows']
                reference=out/('a' if case=='a' else 'registered/b')/'reload.csv'
                require(digest(reference)==evidence['predictions'],'Changed historical validation file')
                designated=read(out/'validation.json')['a']['correct'] if case=='a' else evidence['designated_correct']
                source_run=evidence
            solution_folder=export(native,work/('export_'+case))
            solution=folder/'solution'
            shutil.move(solution_folder,solution)
            manifest=read(solution/'bundle_manifest.json')
            require(manifest['model_sha256']==model,'Unexpected estimator')
            shutil.copyfile(ROOT/'unified_agent.py',solution/'src/unified_agent.py')
            require(source_identity(solution/'src')==solver,'Prediction runtime differs')
            manifest['files']['src/unified_agent.py']=digest(solution/'src/unified_agent.py')
            (solution/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
            copy_resources(manifest['external_resources'],folder,out/'notices')
            shutil.copyfile(solution/'requirements.txt',folder/'requirements.txt')
            (folder/'predict.py').write_text(LAUNCH,encoding='utf-8')
            election=read(original/'run/agent/final_selection.json')
            save(folder/'election.json',election)
            save(folder/'training_scope.json',dict(arm=arm,rows=rows,evidence=source_run))
            if augmented:
                save(folder/'training_log.json',refit['final_fits'])
            else:
                trials=read(native/'history.json')
                save(folder/'training_log.json',[{k:t[k] for k in ('trial','candidate','split_id','status','primary_metric',
                      'search_score','train_score','training_seconds','compute') if k in t} for t in trials])
            verified[task]=dict(selected_model=model,rows=evidence['rows'],predictions=digest(reference),
                                training_rows=rows,designated_correct=designated,
                                recorded_fits=refit['actual_classifier_fits'] if augmented else evidence['fits'],
                                elected_members=evidence['members'],resources=[(r['model'],r.get('revision')) for r in manifest['external_resources']])
        (tree/'README.md').write_text(
            '# 两个课程任务的冻结预测器\n\n'+('仅用原训练集拟合。' if arm=='train_only' else
            '在原训练集和指定验证集上重训，沿用原选择及全部权重。这组没有独立的指定验证成绩。')
            +'\n\n进入题目目录，安装 requirements.txt 所列依赖后，运行 '
            '\x60python -I predict.py --features features.zip --output predictions.csv\x60。'
            '两个目录内的模型使用同一份整合源码，随包提供冻结权重和许可。'
            'election.json 保存原自主训练的选择，training_scope.json 说明本包实际使用的数据。\n',encoding='utf-8')
        staged=work/archive.name
        with zipfile.ZipFile(staged,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=1) as z:
            for p in sorted(tree.rglob('*')):
                if p.is_file():
                    z.write(p,p.relative_to(tree))
        relocated=work/'new_location'
        with zipfile.ZipFile(staged) as z:
            z.extractall(relocated)
        receiver=work/'receiver'
        receiver.mkdir()
        for case,task in [('a','Task1'),('b','Task2')]:
            result=receiver/(task+'.csv')
            subprocess.run([sys.executable,'-I',str(relocated/task/'predict.py'),'--features',str(out/case/'queries.zip'),
                            '--output',str(result)],cwd=receiver,env=independent_env(),stdin=subprocess.DEVNULL,
                           check=True,timeout=600)
            entry=verified[task]
            with result.open(newline='',encoding='utf-8-sig') as stream:
                indices=[r['row_id'] for r in csv.DictReader(stream)]
            require(digest(result)==entry['predictions'] and indices==list(map(str,range(entry['rows']))),
                    'Package output differs from its original native predictor '+task)
        record=dict(arm=arm,archive=archive.name,archive_sha256=digest(staged),bytes=staged.stat().st_size,solver=solver,
                    verified=verified,independent_offline_reload=True,new_fits_during_export=0,model_api=0,official_test_opened=False)
        os.replace(staged,archive)
        save(receipt,record)
        print(json.dumps(record),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('out','text-election'):
        parser.add_argument('--'+name,type=lambda v:Path(v).resolve(),required=True)
    parser.add_argument('--arm',required=True,choices=tuple(FILENAMES))
    args=parser.parse_args()
    package(args.out,args.text_election,args.arm)
