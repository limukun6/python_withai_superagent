"""Publish compact integration evidence; exclude all private ledger, task data and models."""
from collections import Counter
from pathlib import Path
import json
import shutil
from scripts.a_batch_common import save,hashes
from scripts.run_a10_real_task_pilot import read,sha

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parents[1]


def main():
    output=ROOT/'research/a125_a129_summary';output.mkdir(parents=True,exist_ok=False)
    frozen=hashes(ROOT)
    save(output/'protocol.json',dict(batch='A125-A129-summary',source_files=frozen,runner_sha256=sha(Path(__file__)),
        provider='5329503282245a5e8d60f42af46e0ae5f32bf187',source_commit='87634a5',actual_classifier_fits=0,real_API_requests=0))
    names=['a125_B_patches','a125_source_audit','a126_joint_checks','a126_joint_checks_v2','a126_menu_entry',
           'a126_menu_entry_v2','a126_e5_assets','a127_joint_preflight','a127_joint_run','a128_H1_snapshot',
           'a128_H1_local_preflight','a128_H1_local','a129_B_ledger_return']
    allowed=['protocol.json','results.json','completion.json','assets.json','regression.log',
             'experiment_snapshot.json','registration.json','ledger_owner.json']
    copied=[]
    for name in names:
        source=BASE/'runs'/name;target=ROOT/'research'/name;target.mkdir(exist_ok=False)
        for filename in allowed:
            path=source/filename
            if path.exists():shutil.copy2(path,target/filename);copied.append(dict(path=(target/filename).relative_to(ROOT).as_posix(),sha256=sha(path)))
    h=read(BASE/'runs/a128_H1_local/completion.json');anon=read(BASE/'runs/a127_joint_run/completion.json')
    ar=read(BASE/'runs/a127_joint_run/results.json');ledger=read(BASE/'runs/a129_B_ledger_return/results.json')
    new_fits=h['actual_classifier_fits']+anon['actual_classifier_fits']
    if new_fits!=28 or anon['A_cumulative_fits_after']!=1566:raise ValueError('Aggregate count mismatch')
    models=[]
    for directory in [BASE/'runs/a127_joint_run/run',*[BASE/'runs/a128_H1_local'/s/'run' for s in ('61411','61421')]]:
        manifest=read(directory/'run_manifest.json');normalized={k.removeprefix('superagent/'):v for k,v in frozen.items()}
        if manifest['source_files']!=normalized:raise ValueError('Frozen source differs from trained model')
        selection=read(directory/'final_selection.json')
        if sha(directory/selection['model_file'])!=selection['model_sha256']:raise ValueError('Trained model bytes changed')
        models.append(dict(run_dir=str(directory),model_sha256=selection['model_sha256'],selected=selection['family']))
    result=dict(new_classifier_fits=new_fits,A_cumulative_fits_after=1566,real_API_requests=0,new_API_cost_rmb=0,
        tests_passed=read(BASE/'runs/a126_joint_checks_v2/results.json')['tests_run'],
        default_menus_unchanged=read(BASE/'runs/a126_menu_entry_v2/results.json')['default_menus_unchanged'],
        anonymous=dict(fits=6,candidates=ar['menu_count'],e5_tried=ar['e5_tried'],selected=ar['selection']['family'],
            native_relocated_portable_identical=ar['native_relocated_portable_identical'],
            actual_ensemble_selected=ar['actual_ensemble_selected'],elapsed_seconds=anon['elapsed_seconds']),
        H1=dict(local_paths=2,fits=22,elapsed_seconds=h['elapsed_seconds'],outer_scores_computed=0,paid_R0_paths=0),
        E5_public_download=dict(requests=34,seconds=read(BASE/'runs/a126_e5_assets/completion.json')['elapsed_seconds']),
        real_ledger_received=True,ledger_sha256=ledger['ledger_sha256'],project_summary=ledger['project_summary'],
        old_requests_preserved=335,old_transitions_preserved=503,ledger_owner='A',new_paid_calls_permitted=False,
        B_confirmed_current_configuration=True,B_H2_status='not run; B continues H2/Released Task2; no active B queue',
        excluded_next_iteration='924e88a optional on-demand inference',models=models,
        source_equals_all_model_training_sources=True,copied_evidence=copied,
        limitations=['Synthetic autonomous run is engineering evidence, not real-task accuracy gain',
            'No actual ensemble selected in A127; no new local real ensemble-delivery performance proof',
            'H1 local models sealed but R0 not run; outer scores remain unopened',
            'Original C1 and B 689/940 retain their original version/data identities'])
    save(output/'results.json',result)
    save(output/'completion.json',dict(status='archived',actual_classifier_fits=0,real_API_requests=0,A_cumulative_fits_after=1566))
    print(json.dumps({k:result[k] for k in ('new_classifier_fits','A_cumulative_fits_after','real_API_requests','tests_passed','anonymous','H1','real_ledger_received','new_paid_calls_permitted')}),flush=True)


if __name__=='__main__':main()
