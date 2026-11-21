"""Archive selected text/JSON evidence only; no data, models, keys or SQLite."""
import argparse
from pathlib import Path
import shutil

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import sha


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    inventory=[]
    def copy(run,name):
        source=a.runs/run/name
        if not source.is_file():return
        target=a.output/run/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        inventory.append(dict(file=target.relative_to(a.output).as_posix(),sha256=sha(target),bytes=target.stat().st_size))
    checks=['a12_interface_checks','a12_interface_checks_fixed','a14_symmetric_input',
        'a14_symmetric_input_counted','a15_final_entry_regression','a15_final_entry_regression_project_gate']
    for run in checks:
        for name in ('protocol.json','completion.json','tests.log'):copy(run,name)
    root=a.runs/'a12_dynamic_flash_acceptance'
    for name in ('protocol.json','completion.json','results.json','physical_fits.jsonl'):copy(root.name,name)
    for case in root.glob('ab-*'):
        if not case.is_dir():continue
        for name in ('run_manifest.json','run_outcome.json','actions.json','history.json','events.jsonl',
            'classifier_fits.json','final_selection.json','confirmation.json'):
            copy(root.name,case.name+'/'+name)
    for file in root.glob('*_contexts.json'):copy(root.name,file.name)
    for run in ('a13_data_registration','a13_data_registration_network','a13_H1_proxy','a13_H1_publisher',
                'a13_H2_proxy','a13_holdout_packages','a13_photo_pilot','a16_blocked_preflight_fixture'):
        for name in ('protocol.json','registration.json','completion.json','results.json','export_verification.json'):
            copy(run,name)
    for name in ('run_manifest.json','run_outcome.json','actions.json','history.json','classifier_fits.json','final_selection.json'):
        copy('a13_photo_pilot','official_run/'+name)
    save(a.output/'resource_corrections.json',dict(
        a14_symmetric_input=dict(original_completion_fit_count=0,corrected_actual_classifier_fits=1,
            reason='Existing semantics test performs one DummyClassifier.fit; counted rerun also performs one'),
        slow_H2_direct_download=dict(directory='a13_H2_registration',status='interrupted_slow_download',
            model_fits=0,real_api_requests=0,partial_archive_retained_locally=True),
        scope='A12 HTTP calls are simulated; no synthetic ledger is real historical accounting'))
    save(a.output/'inventory.json',dict(files=inventory,bytes=sum(r['bytes'] for r in inventory),
        note='Hashes identify original local bytes; Git may normalize CRLF. No raw data/model/key/SQLite included.'))
    print(dict(files=len(inventory),bytes=sum(r['bytes'] for r in inventory)))
