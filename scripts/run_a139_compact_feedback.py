"""Freeze three real two-trial histories and test lossless compact evidence with no API/fits."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from scripts.a_batch_common import save,hashes
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a133_routing_verification import GUARD

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parents[1]
SOURCES=[('coupon','coupon',BASE/'runs/a135_task1_current_v2/run'),
         ('sms','sms',BASE/'runs/a122_acceptance_continued/sms/run'),
         ('flowers','flowers',BASE/'runs/a122_acceptance_continued/flowers/run')]


def wire_size(v):return len(json.dumps(v,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic();source=hashes(ROOT)
    save(out/'protocol.json',dict(batch='A139',source_files=source,runner_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT/'docs/batches/A139_compact_projection_protocol_20261124_zh.md'),
         source_states={n:dict(path=str(s/'state_3.json'),sha256=sha(s/'state_3.json')) for n,_,s in SOURCES},
         maximum_fits=0,maximum_API_calls=0,starting_A_fits=1614,minimum_checked_overhead_reduction=.20))
    try:
        env={**os.environ,'SUPERAGENT_OFFLINE':'1','PYTHONIOENCODING':'utf-8','PYTHONPATH':str(ROOT)}
        tests=['test_a139_compact_feedback','test_ab_flash','test_ab_encoder_routing','test_a137_numeric_kernel',
               'test_task_goal','test_ab_registered_rows','test_a120_parameter_semantics','test_a130_accepted_history']
        code=GUARD.split('target=sys.argv[1]')[0]+"\nimport unittest\ns=unittest.TestLoader().loadTestsFromNames("+repr(['tests.'+t for t in tests])+")\nr=unittest.TextTestRunner(verbosity=2).run(s)\nraise SystemExit(not r.wasSuccessful())\n"
        with (out/'tests.log').open('x',encoding='utf-8') as stream:
            result=subprocess.run([sys.executable,'-c',code],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=120)
        if result.returncode:raise RuntimeError('Compact/regression tests failed; see tests.log')
        from scripts import a38_predict_guard
        from superagent.ab_flash import context_for
        from superagent.compact_feedback import expand_context
        records=[];sources=[]
        for name,scope,run in SOURCES:
            original=read(run/'state_3.json');caps=read(run/'run_manifest.json')['capabilities']
            state={k:deepcopy(original[k]) for k in ('task','history','available_candidates','budget')}
            if len(state['history'])!=2:raise ValueError('Registered two-trial state differs')
            state['budget'].update(llm_request_limit=3,used_llm_requests=0)
            ids=[c['id'] for c in state['available_candidates']]
            folder=out/name;folder.mkdir()
            views={'raw':context_for(state,caps,ids,False,'calibrated-v2'),
                   'full':context_for(state,caps,ids,True,'calibrated-v2'),
                   'compact':context_for(state,caps,ids,True,'calibrated-v2','compact')}
            if expand_context(views['compact'])!=views['full']:raise ValueError('Real context round-trip differs')
            for k,v in views['raw'].items():
                if views['full'][k]!=v or views['compact'][k]!=v:raise ValueError('Common evidence altered: '+k)
            save(folder/'state.json',state);save(folder/'capabilities.json',caps)
            for view,v in views.items():save(folder/(view+'.json'),v)
            sizes={k:wire_size(v) for k,v in views.items()};overhead=sizes['full']-sizes['raw']
            records.append(dict(task=name,scope=scope,source_state_sha256=sha(run/'state_3.json'),
                 sizes_utf8=sizes,checked_overhead=overhead,saved_bytes=sizes['full']-sizes['compact'],
                 exact_round_trip=True,common_fields_identical=True,
                 original_history_was_local=True,natural_LLM_error_correction_tested=False))
            sources.append(dict(task=name,scope=scope,state_sha256=sha(folder/'state.json'),
                 capabilities_sha256=sha(folder/'capabilities.json'),
                 views_sha256={v:sha(folder/(v+'.json')) for v in views}))
        saved=sum(r['saved_bytes'] for r in records);overhead=sum(r['checked_overhead'] for r in records)
        eligible=all(r['saved_bytes']>=0 for r in records) and overhead>0 and saved/overhead>=.20
        if hashes(ROOT)!=source:raise ValueError('Source changed')
        save(out/'frozen_inputs.json',sources)
        save(out/'results.json',dict(tasks=records,total_saved_bytes=saved,total_checked_overhead=overhead,
             checked_overhead_reduction=saved/overhead if overhead else None,eligible_for_A140=eligible,
             actual_token_or_cost_saving_measured=False,default_feedback_view='full'))
        save(out/'completion.json',dict(status='passed',eligible_for_A140=eligible,actual_classifier_fits=0,
             real_API_requests=0,A_cumulative_fits_after=1614,source_unchanged=True,elapsed_seconds=time.monotonic()-started,
             protocol_sha256=sha(out/'protocol.json'),results_sha256=sha(out/'results.json')))
        print(json.dumps(read(out/'results.json')),flush=True)
    except Exception as e:
        save(out/'completion.json',dict(status='failed',error=str(e),actual_classifier_fits=0,real_API_requests=0,
             elapsed_seconds=time.monotonic()-started));raise


if __name__=='__main__':main()
