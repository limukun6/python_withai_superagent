"""Current terminal AB handoffs, B stop acceptance and corrupted-evidence refusal, no models/HTTP."""
import argparse
from pathlib import Path
import shutil

from scripts import a38_predict_guard
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.ab_run import sources
from superagent.ab_scheduler import next_action
from tools.export_b_handoff import export


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('real-runs','mock-run','output','example'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);frozen=sources();results=[]
    save(root/'protocol.json',dict(batch='A56-current-handoff',source_files=frozen,runner_sha256=sha(Path(__file__)),
        exporter_sha256=sha(Path(__file__).resolve().parents[1]/'tools/export_b_handoff.py'),actual_classifier_fit_limit=0,
        real_API_limit=0,only_training_partition_labels=True))
    cases=[(name,args.real_runs/(name+'_complete')) for name in ('coupon','preference','sms','photo','bean')]
    cases.extend([('partial',args.real_runs/'coupon_fault'),('anonymous_fixture',args.mock_run/'raw-history_ok')])
    for name,directory in cases:
        state=export(directory);manifest=read(directory/'run_manifest.json')
        action=next_action(state,capabilities=manifest['capabilities'])
        if action['action']!='stop' or state['provenance']['executable_for_continuation'] or 'training_profile' not in state:
            raise ValueError('Terminal snapshot allowed continuation or lost its profile')
        save(root/(name+'.json'),state)
        results.append(dict(case=name,history_rows=len(state['history']),prediction_records=len(state['prediction_evidence']),
            completed_classifier_fits=state['budget']['completed_classifier_fits'],llm_requests=state['budget']['used_llm_requests'],
            training_profile_present=True,B_returned='stop',exported_state_sha256=sha(root/(name+'.json'))))
        if name=='anonymous_fixture':
            state['provenance'].update(source_fixture_only=True,paid_ledger_eligible=False)
            args.example.parent.mkdir(parents=True,exist_ok=True)
            if args.example.exists():raise FileExistsError('Do not overwrite an agreed interface example')
            save(args.example,state)
    damaged=root/'damaged';damaged.mkdir()
    source=args.mock_run/'raw-history_ok'
    for file in source.iterdir():
        if file.is_file() and (file.suffix=='.json' or file.name in ('primary.joblib','confirmation_1.joblib','confirmation_2.joblib')):
            shutil.copy2(file,damaged/file.name)
    evidence=read(damaged/'prediction_evidence.json');evidence[0]['prediction_sha256']='0'*64;save(damaged/'prediction_evidence.json',evidence)
    try:export(damaged)
    except ValueError as error:results.append(dict(case='corrupted_prediction',status='rejected',error=str(error)))
    else:raise ValueError('Corrupted prediction was exported')
    save(root/'results.json',results);save(root/'completion.json',dict(status='passed',terminal_snapshots=7,corrupted_evidence_rejected=True,
        actual_classifier_fits=0,real_api_requests=0,source_unchanged=sources()==frozen,model_deserialization=False))
    print(read(root/'completion.json'))
