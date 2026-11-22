"""D3 artifact/remote inventory and existing no-fit regression; never a new sealed evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a29_choice_matrix import launch
from superagent.ab_run import sources

ROOT=Path(__file__).resolve().parents[1]


def main(a):
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);frozen=sources();start=time.monotonic();results=[]
    runners=['run_a1_semantics_batch.py','run_a1_evidence_batch.py','run_a05_task_contract_batch.py','run_a06_objective_flow_batch.py']
    save(root/'protocol.json',dict(batch='A64',source_files=frozen,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A64_D3_acceptance_audit_plan.md'),
        invoked_scripts={n:sha(ROOT/'scripts'/n) for n in runners},real_api_requests=0,actual_classifier_fits=0,
        known_cases_reused=True,new_sealed_evidence=False,formal_test_or_H_opened=False))
    try:
        refs=subprocess.check_output(['git','for-each-ref','--format=%(refname:short)|%(objectname)|%(committerdate:iso8601)|%(subject)',
                                     'refs/remotes'],cwd=ROOT).decode('utf-8').splitlines()
        save(root/'remote_refs.json',[dict(zip(('ref','sha','commit_date','subject'),line.split('|',3))) for line in refs])
        commands=[
            ('semantics',[sys.executable,str(ROOT/'scripts'/runners[0]),'--phase','fixed','--output',str(root/'semantics')]),
            ('evidence',[sys.executable,str(ROOT/'scripts'/runners[1]),'--phase','fixed','--output',str(root/'evidence')]),
            ('task_contract',[sys.executable,str(ROOT/'scripts'/runners[2]),'--source-root',str(ROOT),'--output',str(root/'task_contract'),
                '--subset','all','--cases-file',str(root/'original_a05_cases.json'),'--write-cases']),
            ('objective_flow',[sys.executable,str(ROOT/'scripts'/runners[3]),'--source-root',str(ROOT),'--output',str(root/'objective_flow')])]
        for name,cmd in commands:
            if sources()!=frozen:raise ValueError('Source changed before no-fit check')
            code,seconds=launch(cmd,root/(name+'.log'),{**os.environ,'SUPERAGENT_OFFLINE':'1','PYTHONIOENCODING':'utf-8'},120)
            completion=read(root/name/'completion.json')
            results.append(dict(suite=name,process_seconds=seconds,returncode=code,completion=completion,
                results_sha256=sha(root/name/'results.json')));save(root/'results.json',results)
            if code or completion['passed']!=completion['cases'] or completion.get('classifier_fits')!=0:
                raise ValueError('Known no-fit check failed: '+name)
        evidence_paths=[
            'research/a1_semantics_guard/fixed/completion.json','research/a1_evidence_binding/fixed/completion.json',
            'research/a_window_20261121/a05_fixed_reserved/completion.json','research/a_window_20261121/a06_fixed/completion.json',
            'research/b1_ab_handoff/accepted/completion.json','research/b1_confirmation_budget/completion.json',
            'research/b1_bridge_1121/completion.json','research/a50_a59_final_round/a55_fresh_answers/completion.json',
            'research/a50_a59_final_round/a55_fresh_evaluation/completion.json',
            'research/a47_catalog_Q/completion.json','research/a60_a63_finalists_auc/a60_matched_finalists/completion.json',
            'research/a60_a63_finalists_auc/a62_auc_official_loop/completion.json']
        save(root/'historical_evidence_inventory.json',[dict(path=p,sha256=sha(ROOT/p),completion=read(ROOT/p)) for p in evidence_paths])
        total=sum(r['completion']['cases'] for r in results)
        save(root/'completion.json',dict(status='passed',known_check_entries=total,passed=total,
            new_sealed_evaluation=False,actual_classifier_fits=0,real_api_requests=0,tracked_A_fits_unchanged=999,
            source_unchanged=sources()==frozen,elapsed_seconds=time.monotonic()-start,
            simulated_worker_calls=sum(r['completion'].get('simulated_worker_calls',0) for r in results)))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),completed_suites=len(results),
            actual_classifier_fits=0,real_api_requests=0,automatic_retry=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
