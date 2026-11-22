"""One bounded official CLI loop plus same-observation scope rejection checks."""
import argparse
from copy import deepcopy
import os
from pathlib import Path
import sys
import time
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a29_choice_matrix import launch,physical
from scripts.run_b1_ab_acceptance import packages
from superagent.ab_run import sources
from superagent.hypothesis_agent import check_expectation
from superagent.prediction_evidence import fingerprint

ROOT=Path(__file__).resolve().parents[1]


def main(a):
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);frozen=sources();start=time.monotonic()
    save(root/'protocol.json',dict(batch='A68',source_files=frozen,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A68_scope_official_acceptance_plan.md'),maximum_classifier_fits=3,real_api_requests=0,
        starting_tracked_A_fits=1022,scope='Synthetic official CLI and controlled scope counterexamples, not task performance'))
    try:
        packages(root);directory=root/'run';observer=ROOT/'scripts/a29_worker_observer'
        env={**os.environ,'AB_A29_OBSERVE_FITS':'1','AB_A29_FIT_LOG':str(root/'physical_fits.jsonl'),
            'AB_A29_RUN_NAME':'scope_acceptance','PYTHONPATH':os.pathsep.join([str(observer),str(ROOT)]),
            'SUPERAGENT_OFFLINE':'1','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2','PYTHONIOENCODING':'utf-8'}
        code,seconds=launch([sys.executable,'-m','superagent','run',str(root/'train.zip'),'--run-dir',str(directory),
            '--scheduler','ab-v1','--portfolio','ab-v1','--fit-limit','3','--seed','707','--time-limit','240','--trial-timeout','60'],root/'train.log',env,300)
        outcome=read(directory/'run_outcome.json')
        if code or not outcome['requested_planner_completed'] or len(physical(root))!=3 or outcome['classifier_fit_accounting']['completed_classifier_fits']!=3:
            raise ValueError('Official scope acceptance failed; no retry')
        history=read(directory/'history.json');evidence=read(directory/'prediction_evidence.json')
        for row in history:
            p=next(p for p in evidence if p['trial']==row['trial'])
            if any(row[k]!=p[k] for k in ('evaluation_payload_sha256','label_sha256')) or row['row_ids_sha256']!=fingerprint(p['row_ids']):
                raise ValueError('Official measurement identities do not match trusted evidence')
        if len(history)!=2:raise ValueError('Expected two real completed primary observations')
        before,after=history;proposal=dict(expected_change=dict(metric='search_score',reference_trial=before['trial'],direction='increase',min_delta=0))
        base=check_expectation(proposal,after,[before])
        if base['status'] not in ('supported','not_supported') or not base['comparison_scope']['provenance_complete']:
            raise ValueError('Valid bound comparison was not evaluated')
        checks=[]
        for field,value in (('split_id','confirmation_1'),('evaluation_payload_sha256','b'*64),
                             ('label_sha256','c'*64),('primary_metric','macro_f1'),('row_ids_sha256',None)):
            altered=deepcopy(after)
            if value is None:del altered[field]
            else:altered[field]=value
            result=check_expectation(proposal,altered,[before])
            if result['status']!='not_evaluable':raise ValueError('Explicit incomparable observation was confirmed')
            checks.append(dict(field=field,status=result['status'],reason=result['reason']))
        code,_=launch([sys.executable,'-m','scripts.a38_predict_guard',str(root/'test.zip'),'--run-dir',str(directory),
            '--output',str(root/'predictions.csv')],root/'predict.log',{**env,'AB_A29_RUN_NAME':'inference_only'},60)
        if code or len(physical(root))!=3:raise ValueError('Prediction performed training or failed')
        save(root/'results.json',dict(outcome=outcome,valid_comparison=base,controlled_conflicts=checks,training_process_seconds=seconds))
        save(root/'completion.json',dict(status='passed',actual_classifier_fits=3,real_api_requests=0,tracked_A_fits_after=1025,
            official_bound_observations=2,controlled_scope_rejections=len(checks),source_unchanged=sources()==frozen,
            elapsed_seconds=time.monotonic()-start,formal_test_or_H_opened=False))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=len(physical(root)),real_api_requests=0,automatic_retry=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
