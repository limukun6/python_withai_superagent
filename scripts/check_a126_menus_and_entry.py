"""Compare seven default menus and exercise new scheduler fallbacks before Executor construction."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from scripts.a_batch_common import save,hashes
from scripts.run_a10_real_task_pilot import read,sha
from scripts.audit_a111_B_verified import MENU_PROBE

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parents[1]
BOUNDARY='''
import json,sys
from pathlib import Path
from unittest.mock import patch
from scripts import a38_predict_guard
from superagent import ab_run
from superagent.cli import build_parser
from tools.ab_registered_run import run
class StopBeforeTraining(BaseException):pass
def stop(*a,**k):raise StopBeforeTraining()
cfg=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
with patch.object(ab_run,'Executor',side_effect=stop):
    try:
        if cfg.get('manifest'):run(Path(cfg['manifest']),cfg['manifest_sha256'],cfg['command'])
        else:
            args=build_parser().parse_args(cfg['command']);args.handler(args)
    except StopBeforeTraining:print('Reached Executor boundary; no fits performed')
    else:raise ValueError('Expected executor boundary')
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();source=hashes(ROOT)
    save(a.output/'protocol.json',dict(batch='A126-menu-entry',source_files=source,runner_sha256=sha(Path(__file__)),
        maximum_classifier_fits=0,real_API_requests=0,boundary='before Executor construction',outer_labels_accessed=False))
    try:
        env={**os.environ,'SUPERAGENT_MINILM_WEIGHTS':str(BASE/'runs/a82_encoder_assets/minilm'),
            'SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS':str(BASE/'runs/a82_encoder_assets/mobilenet_v3_small-047dcff4.pth'),
            'SUPERAGENT_DEEP_MINILM_WEIGHTS':str(BASE/'runs/a113R_deep_assets/minilm_l12'),
            'SUPERAGENT_E5_SMALL_WEIGHTS':'','SUPERAGENT_BGE_SMALL_WEIGHTS':'',
            'SUPERAGENT_AB_PREFERENCE_CATALOG':'standard','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
            'SUPERAGENT_OFFLINE':'1','PYTHONIOENCODING':'utf-8'}
        env.pop('PYTHONPATH',None)
        metadata=ROOT/'research/a111_B_verified_intake/task_metadata.json';menus=[]
        for folder in (a.baseline,ROOT):
            result=subprocess.run([sys.executable,'-c',MENU_PROBE,str(metadata)],cwd=folder,env=env,
                capture_output=True,text=True,encoding='utf-8',timeout=120)
            if result.returncode:raise RuntimeError(result.stderr[-1000:])
            menus.append(json.loads(result.stdout))
        if menus[0]!=menus[1]:raise ValueError('Default menus changed')
        paths={'coupon':BASE/'runs/a122_prepared_v2/coupon/anonymous.zip',
               'sms':BASE/'runs/a122_prepared_v2/sms/anonymous.zip',
               'flowers':BASE/'runs/a122_prepared_v2/flowers/anonymous.zip',
               'H1_61411':BASE/'runs/a114_H1_inputs/61411/train.zip',
               'H1_61421':BASE/'runs/a114_H1_inputs/61421/train.zip'}
        entries=[]
        for name,path in paths.items():
            cfg=dict(command=['run',str(path),'--run-dir',str(a.output/name),'--planner','local',
                '--scheduler','ab-adaptive-complement','--portfolio','ab-v1','--preference-catalog','deep-e5-familiarity',
                '--cover-untried-encoder','--fit-limit','10','--seed','707','--trial-timeout','300','--time-limit','1200'])
            if name.startswith('H1_'):
                manifest=path.parent/'inner_splits.json';cfg.update(manifest=str(manifest),manifest_sha256=sha(manifest))
            config=a.output/(name+'.json');save(config,cfg)
            result=subprocess.run([sys.executable,'-c',BOUNDARY,str(config)],cwd=ROOT,env=env,
                capture_output=True,text=True,encoding='utf-8',timeout=180)
            (a.output/(name+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
            if result.returncode:raise RuntimeError('Entry failed before executor: '+name)
            manifest=read(a.output/name/'run_manifest.json')
            if manifest['complement_active'] or manifest['adaptive_opening_active']:
                raise ValueError('Preference combination became active for a non-preference input')
            if any(c.get('swap_label_rule')!='preserves' for c in manifest['capabilities'].values()) and name.startswith('H1_'):
                raise ValueError('Directional method leaked into H1')
            entries.append(dict(task=name,input_kind=manifest['task_contract']['kind'],
                menu=len(manifest['capabilities']),complement_active=False,reached_executor_without_fit=True))
        if hashes(ROOT)!=source:raise ValueError('Source changed')
        save(a.output/'results.json',dict(default_menus_unchanged=True,default_task_count=len(menus[0]),entries=entries))
        save(a.output/'completion.json',dict(status='passed',actual_classifier_fits=0,real_API_requests=0,
            source_unchanged=True,A_cumulative_fits_after=1538,elapsed_seconds=time.monotonic()-start))
        print(json.dumps(dict(default_tasks=len(menus[0]),entry_boundaries=len(entries),fits=0,API=0)),flush=True)
    except Exception as error:
        save(a.output/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,real_API_requests=0));raise


if __name__=='__main__':main()
