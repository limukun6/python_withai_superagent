"""Open the held labels only after both frozen transfer models pass identity checks."""
import argparse
import csv
import json
from pathlib import Path
import shutil
import sys
import time

import joblib
import numpy as np

from scripts.a_batch_common import hashes,save
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a122_current_agent_acceptance import child,environment,physical

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('prepared','training','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);began=time.monotonic();source=hashes(ROOT)
    save(out/'protocol.json',dict(batch='A144',source_files=source,runner_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT/'docs/batches/A142_A144_new_numeric_transfer_plan_20261124_zh.md'),
         training_completion_sha256=sha(a.training/'completion.json'),maximum_fits=0,maximum_API_requests=0,
         scope='Supplemental frozen real-task transfer, one grouped split; not H2 or final course test'))
    results=[]
    try:
        from scripts import a38_predict_guard
        done=read(a.training/'completion.json');prepared=read(a.prepared/'results.json')
        if done['status']!='all_models_sealed' or prepared['source_files']!=source:
            raise ValueError('Both same-source models must be sealed before held labels open')
        for arm in ('L0','EC0'):
            run=a.training/arm/'run';manifest=read(run/'run_manifest.json');selection=read(run/'final_selection.json')
            if manifest['capabilities']!=prepared['capabilities'] or manifest['source_files']!={k.removeprefix('superagent/'):v for k,v in source.items()}:
                raise ValueError('Frozen source or common candidate identity differs')
            if sha(run/selection['model_file'])!=selection['model_sha256'] or read(a.training/arm/'completion.json')['status']!='sealed':
                raise ValueError('Original model completion or seal differs')
        for split in ('primary','confirmation_1','confirmation_2'):
            for ext in ('.joblib','_split.json'):
                if sha(a.training/'L0/run'/(split+ext))!=sha(a.training/'EC0/run'/(split+ext)):
                    raise ValueError('Matched actual split files differ')
        held_file=a.prepared/'private/holdout.joblib'
        if sha(held_file)!=prepared['holdout_sha256'] or sha(a.prepared/'query.zip')!=prepared['query_archive_sha256']:
            raise ValueError('Sealed evaluation bytes differ')
        held=joblib.load(held_file);env=environment(out)
        for arm in ('L0','EC0'):
            folder=out/arm;folder.mkdir();run=a.training/arm/'run';query=a.prepared/'query.zip'
            selected=read(run/'final_selection.json');fits=[r for r in physical(a.training) if r['run_name']==arm]
            child([sys.executable,'-m','scripts.a38_predict_guard',str(query),'--run-dir',str(run),'--output',str(folder/'native.csv')],folder/'native.log',env)
            moved=folder/'relocated';moved.mkdir()
            for n in ('task_spec.json','final_selection.json',selected['model_file']):shutil.copy2(run/n,moved/n)
            child([sys.executable,'-m','scripts.a38_predict_guard',str(query),'--run-dir',str(moved),'--output',str(folder/'relocated.csv')],folder/'relocated.log',env)
            child([sys.executable,'-m','tools.export_model_bundle','--run-dir',str(run),'--output',str(folder/'portable')],folder/'bundle.log',env)
            child([sys.executable,'-I',str(folder/'portable/bundle/predict.py'),'--features',str(query),'--output',str(folder/'isolated.csv')],folder/'isolated.log',env)
            if len({(folder/n).read_bytes() for n in ('native.csv','relocated.csv','isolated.csv')})!=1:raise ValueError('Three independent predictions differ')
            code="import runpy,sys;runpy.run_path(sys.argv[1]);sys.argv=sys.argv[2:];runpy.run_module('scripts.a07_verify_export',run_name='__main__')"
            child([sys.executable,'-c',code,str(ROOT/'scripts/a38_predict_guard.py'),'verify','--source-root',str(ROOT),
                 '--run-dir',str(run),'--holdout',str(held_file),'--predictions',str(folder/'native.csv'),
                 '--output',str(folder/'verification.json')],folder/'verify.log',env)
            verification=read(folder/'verification.json')
            with (folder/'native.csv').open(encoding='utf-8',newline='') as stream:rows=list(csv.DictReader(stream))
            pred=np.asarray([int(r['prediction']) for r in rows]);labels=held['y'];hits=(pred==labels)
            if len(pred)!=prepared['held_rows'] or [int(r['row_id']) for r in rows]!=list(range(len(pred))):raise ValueError('Row identity differs')
            usage=[json.loads(l) for l in (run/'llm_usage.jsonl').read_text(encoding='utf-8').splitlines()]
            calls=[r for r in usage if r.get('event')=='llm_call']
            outcome=read(run/'run_outcome.json')
            from sklearn.metrics import f1_score
            record=dict(arm=arm,rows=len(pred),correct=int(hits.sum()),accuracy=float(hits.mean()),
                 macro_f1=float(f1_score(labels,pred,average='macro',zero_division=0)),candidate=selected['family'],
                 actual_fits=len(fits),API_calls=len(calls),input_tokens=sum(c['input_tokens'] for c in calls),
                 output_tokens=sum(c['output_tokens'] for c in calls),known_cost_rmb=sum(c['course_cost_rmb'] for c in calls),
                 formal_run_seconds=outcome['elapsed_seconds'],API_latency_seconds=sum(c['latency_seconds'] for c in calls),
                 independent_predictions_identical=True,model_sha256=selected['model_sha256'],
                 prediction_sha256=sha(folder/'native.csv'),verification=verification)
            results.append(record);save(out/'checkpoint.json',results);print(json.dumps(record),flush=True)
        def predicted(arm):
            with (out/arm/'native.csv').open(encoding='utf-8',newline='') as stream:return np.asarray([int(r['prediction']) for r in csv.DictReader(stream)])
        old,new=predicted('L0'),predicted('EC0');truth=held['y'];old_hit=old==truth;new_hit=new==truth
        paired=dict(rows=len(truth),changed_predictions=int((old!=new).sum()),newly_correct=int((~old_hit&new_hit).sum()),
             newly_wrong=int((old_hit&~new_hit).sum()),net_correct=int(new_hit.sum()-old_hit.sum()),
             accuracy_delta_pp=float(100*(new_hit.mean()-old_hit.mean())))
        majority=float((truth==prepared['majority_training_label']).mean())
        if physical(out) or hashes(ROOT)!=source:raise ValueError('Evaluation trained classifiers or source changed')
        save(out/'results.json',dict(paths=results,paired=paired,majority_train_label_accuracy=majority,
             actual_classifier_fits=0,real_API_requests=0,one_new_real_task=True,independent_task_count=1,
             isolated_compaction_effect_identified=False,
             scope='Frozen two whole-system configurations; single grouped split, no held-score tuning'))
        save(out/'completion.json',dict(status='passed',actual_classifier_fits=0,real_API_requests=0,
             elapsed_seconds=time.monotonic()-began,source_unchanged=True,
             protocol_sha256=sha(out/'protocol.json'),results_sha256=sha(out/'results.json')))
    except Exception as e:
        save(out/'results.json',dict(paths=results));save(out/'completion.json',dict(status='failed',error=str(e),actual_classifier_fits=0,real_API_requests=0));raise


if __name__=='__main__':main()
