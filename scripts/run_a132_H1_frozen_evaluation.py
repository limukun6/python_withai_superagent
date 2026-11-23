"""Score all four frozen H1 models once, then validate portable delivery without any new training/API."""
import argparse
import csv
from pathlib import Path
import shutil
import sys
import time
from scripts.a_batch_common import save,hashes
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a122_current_agent_acceptance import environment,child

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('local','paid','snapshot','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();source=hashes(ROOT)
    save(a.output/'protocol.json',dict(batch='A132',source_files=source,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A130_A132_H1_paid_continuation_plan_20261123_zh.md'),
        maximum_classifier_fits=0,real_API_requests=0,outer_scoring_requires_all_four_models_frozen=True))
    try:
        matrix={seed+'_'+arm:str(parent/seed) for seed in ('61411','61421') for arm,parent in (('L0',a.local),('R0',a.paid))}
        for name,folder in matrix.items():
            path=Path(folder);seal=read(path/'model_seal.json');done=read(path/'completion.json')
            if done['status']!='sealed' or not seal['complete']:raise ValueError('Planned path not complete: '+name)
        target=a.output/'matrix.json';save(target,matrix);env=environment(a.output)
        child([sys.executable,'-m','scripts.run_a123_H1_frozen','score','--prepared',str(a.snapshot),
            '--matrix',str(target),'--matrix-sha256',sha(target),'--output',str(a.output/'score')],a.output/'score.log',env,600)
        scores=read(a.output/'score/results.json');deliveries=[];comparisons=[]
        for seed in ('61411','61421'):
            features=a.output/'score'/seed/'features.zip'
            for arm in ('L0','R0'):
                folder=a.output/(seed+'_'+arm);folder.mkdir();run=Path(matrix[seed+'_'+arm])/'run'
                selection=read(run/'final_selection.json')
                moved=folder/'relocated';moved.mkdir()
                for file in ('task_spec.json','final_selection.json',selection['model_file']):shutil.copy2(run/file,moved/file)
                child([sys.executable,'-m','scripts.a38_predict_guard',str(features),'--run-dir',str(moved),
                    '--output',str(folder/'relocated.csv')],folder/'relocated.log',env,240)
                child([sys.executable,'-m','tools.export_model_bundle','--run-dir',str(run),'--output',str(folder/'portable')],folder/'bundle.log',env,180)
                bundle=folder/'portable/bundle'
                child([sys.executable,'-I',str(bundle/'predict.py'),'--features',str(features),
                    '--output',str(folder/'isolated.csv')],folder/'isolated.log',env,240)
                original=a.output/'score'/seed/(arm+'.csv')
                if original.read_bytes()!=(folder/'relocated.csv').read_bytes() or original.read_bytes()!=(folder/'isolated.csv').read_bytes():
                    raise ValueError('Independent delivery prediction differs')
                deliveries.append(dict(outer=seed,arm=arm,native_relocated_portable_identical=True,
                    model_sha256=selection['model_sha256'],prediction_sha256=sha(original)))
            rows={}
            for arm in ('L0','R0'):
                with (a.output/'score'/seed/(arm+'.csv')).open(encoding='utf-8',newline='') as stream:rows[arm]=list(csv.DictReader(stream))
            if len(rows['L0'])!=len(rows['R0']):raise ValueError('Paired prediction lengths differ')
            s={r['arm']:r for r in scores if r['outer']==seed}
            comparisons.append(dict(outer=seed,rows=s['L0']['rows'],L0_accuracy=s['L0']['accuracy'],R0_accuracy=s['R0']['accuracy'],
                R0_minus_L0_percentage_points=100*(s['R0']['accuracy']-s['L0']['accuracy']),
                net_correct=s['R0']['correct']-s['L0']['correct'],
                prediction_disagreements=sum(x['prediction']!=y['prediction'] for x,y in zip(rows['L0'],rows['R0'],strict=True))))
        if hashes(ROOT)!=source:raise ValueError('Source changed')
        save(a.output/'results.json',dict(scores=scores,comparisons=comparisons,deliveries=deliveries,
             selection_changed_after_H=False,independent_task_count=1,outer_partitions=2))
        save(a.output/'completion.json',dict(status='completed_frozen_H1',paths=4,actual_classifier_fits=0,real_API_requests=0,
            source_unchanged=True,elapsed_seconds=time.monotonic()-start))
        print(comparisons,flush=True)
    except Exception as error:
        save(a.output/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,real_API_requests=0,
             elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':main()
