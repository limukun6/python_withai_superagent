"""Check frozen scores, live predictions made during the call and the novelty of each proposed experiment."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from scripts.run_conditional_review_study import get_predictions
from scripts.run_proposal_value_study import DELIVERED_CASES, DELIVERED_MODES
from scripts.unified_released_study import ROOT, env, read, require, save, sha


def observe(root,weights):
    """Start alongside the serial paid stage to exercise the normal public predict entry while it waits."""
    for case in DELIVERED_CASES:
        case_dir=root/case
        run=case_dir/DELIVERED_MODES[1]/'run'
        pointer=run/'agent/final_selection.json'
        deadline=time.monotonic()+650
        while time.monotonic()<deadline:
            if pointer.is_file():
                selected=read(pointer)
                if selected.get('model_file')=='core_ready_selected_model.joblib':
                    break
                require(not selected.get('refit'),'Missed protected delivery')
            time.sleep(.25)
        else:
            raise TimeoutError('No protected model appeared on time')
        target=case_dir/'live-check'
        target.mkdir(exist_ok=False)
        output=target/'predictions.csv'
        start=time.time()
        proc=subprocess.run([sys.executable,'-m','unified_agent','predict',str(case_dir/'input.zip'),'--run-dir',str(run),
                             '--output',str(output)],cwd=ROOT,env=env(weights,target/'unexpected_fits.jsonl'),
                            capture_output=True,text=True,timeout=90)
        end=time.time()
        manifest=read(target/'prediction_manifest.json') if output.exists() else {}
        save(target/'observation.json',dict(observed_model_sha256=selected['model_sha256'],same_throughout=selected==read(pointer),
             start_epoch=start,end_epoch=end,return_code=proc.returncode,delivery=manifest.get('delivery_status'),
             sha=manifest.get('prediction_sha256'),rows=manifest.get('prediction_count')))


def verify(root):
    protocol=read(root/'protocol.json')
    frozen=read(root/'frozen_predictions.json')
    results=read(root/'completion.json')
    report={}
    scores_checked=0
    for case in DELIVERED_CASES:
        spec=protocol['inputs'][case]
        case_dir=root/case
        truth=root/'evaluation'/(case+'.npy')
        require(sha(truth)==spec['truth_sha'],'Changed sealed labels')
        y=np.load(truth,allow_pickle=False)
        measure={'accuracy':accuracy_score,'balanced_accuracy':balanced_accuracy_score}[spec['metric']]
        for mode in DELIVERED_MODES:
            scores=results['cases'][case]['arms'][mode]['scores']
            require(set(scores)=={'fourth','seconds30','protected','final'},'Missing delivery')
            for cut,artifact in frozen[case][mode].items():
                predictions=get_predictions(root/artifact['path'],y,artifact['sha'])
                require(abs(measure(y,predictions)-scores[cut])<1e-12,'Independently computed score differs')
                scores_checked+=1

        local,paid=DELIVERED_MODES
        run=case_dir/paid/'run/agent'
        live_dir=case_dir/'live-check'
        live=read(live_dir/'observation.json')
        require(live['return_code']==0 and live['delivery']=='partial' and live['same_throughout'] and
                live['rows']==len(y) and not (live_dir/'unexpected_fits.jsonl').exists(),'Live prediction failed')
        check_sha=frozen[case][local]['protected']['sha']
        require(check_sha==frozen[case][paid]['protected']['sha']==live['sha']==sha(live_dir/'predictions.csv'),
                'Live protected predictions differ from subsequent offline replays')
        arms=results['cases'][case]['arms']
        review=arms[paid]['review']
        require(arms[local]['review']['protected_model']==review['protected_model']==live['observed_model_sha256'],
                'The protected artifacts differed')
        calls=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines()]
        dispatch,=[e for e in calls if e.get('kind')=='llm_request']
        settled,=read(case_dir/paid/'sealed.json')['calls']
        require(dispatch['budget_request_id']==settled['budget_request_id'] and
                datetime.fromisoformat(dispatch['time_utc']).timestamp()<=live['start_epoch']<=live['end_epoch']<=
                datetime.fromisoformat(settled['time_utc']).timestamp(),'Live prediction was outside the actual call')
        history=read(run/'history.json')
        measured={r['candidate'] for r in history if r['status']=='completed'}
        packet=read(run/'experiment_proposal_input.json')
        candidate=review['plan'].get('candidate')
        origin=('keep' if candidate is None else 'already_selected_on_core_route' if candidate in review['core_members']
                else 'same_as_local_extra_experiment' if candidate==review['local_alternative'] else 'other_candidate')
        report[case]=dict(live_rows=len(y),protected_sha=check_sha,live_score=float(measure(y,get_predictions(live_dir/'predictions.csv',y,live['sha']))),
                          seconds_between_live_export_and_response=round(datetime.fromisoformat(settled['time_utc']).timestamp()-live['end_epoch'],3),
                          proposal_identity=origin,proposal_already_tried_at_core_trials=[r['trial'] for r in history if r['status']=='completed' and r['candidate']==candidate],
                          listed_options=len(packet['options']),untried_options_excluding_local_tail=sum(
                              row['name'] not in measured and row['name']!=review['local_alternative'] for row in packet['options']),
                          remaining_fits_after_complete_local_route=16-arms[local]['fits'])
    return dict(independent_snapshot_scores=scores_checked,real_during_call_exports=len(report),cases=report,additional_fits=0,additional_api_calls=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=lambda p:Path(p).resolve())
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output',type=Path,help='Verify completed and sealed observations')
    mode.add_argument('--observe-with-weights',type=lambda p:Path(p).resolve(),
                      help='Run a live prediction while one campus call per task is in progress')
    args=parser.parse_args()
    if args.output is not None:
        save(args.output,verify(args.root))
    else:
        observe(args.root,args.observe_with_weights)
