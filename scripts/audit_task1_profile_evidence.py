"""Cross-check saved development evidence and replay the sealed model without labels."""
import argparse
import json
import time
from pathlib import Path

from scripts.run_task1_profile_screen import digest, save
from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT, env
from scripts.verify_anonymous_planner import execute


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def run(output,private):
    output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    record={}
    fit_total=0
    for tag,cap in (('profile_screen',8),('joint_context',9),('latent_towers',4),('context_capacity',6),('numeric_copy',6)):
        folder=ROOT/'research'/f'task1_{tag}_01'
        completion=read(folder/'completion.json')
        if completion['status']!='complete' or completion['fits']!=cap or completion['api_calls']!=0:
            raise ValueError('Incomplete batch accounting '+tag)
        for row in read(folder/'results.json'):
            base=row.get('baseline') or row.get('tree') or row.get('incumbent')
            for entry in row.values():
                if isinstance(entry,dict) and 'fixed' in entry:
                    if entry['correct']-base['correct']!=entry['fixed']-entry['broken']:
                        raise ValueError('Paired errors did not reconcile '+tag)
        fit_total+=cap
        record[tag]=dict(fits=cap,completion_sha=digest((folder/'completion.json').read_bytes()))
    foundation=read(ROOT/'research/task1_tabpfn_02/completion.json')
    failed=read(ROOT/'research/task1_tabpfn_01/completion.json')
    if (foundation['status']!='complete' or foundation['fit_api_calls']!=4 or
            foundation['classifier_members']!=8 or foundation['api_calls']!=0 or failed['fit_api_calls']!=0):
        raise ValueError('Foundation-model accounting differs')
    record['tabpfn']=dict(fit_calls=4,classifier_members=8,initial_failed_fits=0)
    sealed=read(ROOT/'research/task1_context_agent_01/sealed.json')
    completed=read(ROOT/'research/task1_context_agent_01/completion.json')
    if completed['agent_fits']!=10 or completed['diagnostic_fits']!=0 or completed['api_calls']!=0:
        raise ValueError('Autonomous fit accounting differs')
    model=private/'agent_run/agent/selected_model.joblib'
    selected=read(private/'agent_run/agent/final_selection.json')
    if digest(model.read_bytes())!=sealed['model'] or selected['training_rows']!=8117:
        raise ValueError('Model identity or training membership differs')
    physical=private/'physical.jsonl'
    before=physical.read_bytes()
    if len(before.splitlines())!=10 or len(sealed['workers'])!=10:
        raise ValueError('Worker observer differs')
    record['context_agent']=dict(fits=10,model_sha=sealed['model'],source_at_training=sealed['source'],
                                development_score=completed['scores']['agent'])
    settings=env(private/'unused_weights',physical)
    for key in ('SUPERAGENT_PLANNER_KEY_FILE','SUPERAGENT_PLANNER_LEDGER','SUPERAGENT_TASK_BUDGET_ID',
                'SUPERAGENT_RESEARCH_REFLECTION_OUTBOX'):
        settings.pop(key,None)
    settings['PYTHONUTF8']='1'
    replay=private/output.name/'predictions.csv'
    execute(['predict',private/'features.zip','--run-dir',private/'agent_run','--output',replay],
             private/(output.name+'_reload.log'),settings)
    if digest(replay.read_bytes())!=sealed['predictions']['agent'] or physical.read_bytes()!=before:
        raise ValueError('Independent replay changed predictions or fitted')
    fit_total+=10
    save(output/'completion.json',dict(status='complete',batches=record,
        historical_conventional_fits=fit_total,tabpfn_fit_calls=4,tabpfn_members=8,
        conservative_classifier_units=fit_total+8,new_classifier_fits=0,api_calls=0,
        current_source=source_identity(ROOT),reload_prediction_sha=digest(replay.read_bytes()),
        validation_labels_opened=False,test_members_opened=False,seconds=round(time.monotonic()-start,3)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--private',required=True,type=Path)
    args=parser.parse_args()
    run(args.output,args.private.resolve())
