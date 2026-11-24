"""Archive completed one-task six-arm results; separate reused fits and newly paid requests."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import joblib
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha

PAIRS=[('L1','L0'),('R1','R0'),('E1','E0'),('E0','R0'),('E1','R1'),('R0','L0'),('R1','L1'),('E0','L0'),('E1','L1')]
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    completion=read(a.run/'completion.json')
    if completion['status']!='completed':raise ValueError('Six frozen/scored models required; do not score partial batch')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);truth=[str(v) for v in joblib.load(a.run/'evaluation/coupon/holdout.joblib')['y']]
    arms={};predictions={}
    for row in read(a.run/'results.json'):
        arm=row['arm'];directory=Path(row['run_dir']);folder=out/arm;folder.mkdir();usage=[]
        for line in (directory/'llm_usage.jsonl').read_text(encoding='utf-8').splitlines():
            event=json.loads(line)
            if event.get('event')=='llm_call':usage.append(event)
        for name in ['run_manifest.json','final_selection.json','classifier_fits.json','actions.json']:
            shutil.copyfile(directory/name,folder/name)
        with (a.run/'predictions'/arm/'rows.csv').open(encoding='utf-8',newline='') as stream:output=list(csv.DictReader(stream))
        if [int(r['row_id']) for r in output]!=list(range(len(truth))):raise ValueError('Prediction row identities differ')
        predictions[arm]=[r['prediction'] for r in output]
        arms[arm]=dict(score=row['evaluation']['score'],rows=row['evaluation']['rows'],metric=row['evaluation']['primary_metric'],
            selected=row['outcome']['selected_candidate'],new_classifier_fits=row['new_classifier_fits'],
            reused_classifier_fits=row.get('reused_classifier_fits',0),new_process_seconds=row['process_seconds'],
            reused_reference_process_seconds=row.get('reference_process_seconds'),new_API_requests=row['outcome']['new_api_requests'],
            new_known_estimated_rmb=row['outcome']['known_course_cost_rmb'],input_tokens=sum(r['input_tokens'] for r in usage),
            output_tokens=sum(r['output_tokens'] for r in usage),reused=row['reused'],reload_identical=row['evaluation']['reload_identical'],
            prediction_sha256=sha(a.run/'predictions'/arm/'rows.csv'))
        save(folder/'measured_usage_summary.json',dict(requests=len(usage),input_tokens=arms[arm]['input_tokens'],
            output_tokens=arms[arm]['output_tokens'],known_estimated_rmb=arms[arm]['new_known_estimated_rmb']))
    contrasts=[]
    for left,right in PAIRS:
        lp,rp=predictions[left],predictions[right]
        contrasts.append(dict(left=left,right=right,delta=arms[left]['score']-arms[right]['score'],
            prediction_disagreements=sum(a!=b for a,b in zip(lp,rp,strict=True)),
            left_correct_only=sum(a==y and b!=y for a,b,y in zip(lp,rp,truth,strict=True)),
            right_correct_only=sum(b==y and a!=y for a,b,y in zip(lp,rp,truth,strict=True))))
    save(out/'summary.json',dict(arms=arms,contrasts=contrasts,completion=completion,
        new_input_tokens=sum(r['input_tokens'] for r in arms.values()),new_output_tokens=sum(r['output_tokens'] for r in arms.values()),
        known_incremental_estimated_rmb=sum(r['new_known_estimated_rmb'] for r in arms.values()),
        scope='One known task/development holdout; L1 predeclared reuse and already seen; not cross-task or budget-compliance proof',
        actual_classifier_fits=0,real_api_requests=0))
    for name in ('protocol.json','completion.json','physical_fits.jsonl','all_models_frozen.json'):
        shutil.copyfile(a.run/name,out/name)
    print({name:dict(score=r['score'],new_fits=r['new_classifier_fits'],reused_fits=r['reused_classifier_fits'],requests=r['new_API_requests']) for name,r in arms.items()})
