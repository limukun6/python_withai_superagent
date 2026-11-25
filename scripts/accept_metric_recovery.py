"""Run the official solve/predict commands under controlled, no-network failures."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

from scripts.accept_record_formats import ROOT,read,save


def preference_probe():
    """Send identical, imbalanced rows and a one-fit budget through the real local scheduler."""
    from superagent.experiment_policy import actions
    from superagent.metrics import MetricSpec

    labels = ['a']*9 + ['b']
    predictions = {'majority_only':['a']*10, 'finds_minority':['a']*7+['b']*3}
    capabilities = {name:dict(estimator='linear',representation='numeric',params={},
                              classifier_fits=1,refit_classifier_fits=1) for name in predictions}
    profile = dict(statistics={'train_rows':40},majority_class_fraction=.9,minority_class_fraction=.1)
    result = {}
    for metric_name in ('Accuracy','macro-F1'):
        metric = MetricSpec(metric_name,('a','b'))
        scores = {name:metric.evaluate(labels,values)['score'] for name,values in predictions.items()}
        bank = {name:dict(score=scores[name],labels=labels,predictions=values,
                          probabilities=None,row_key='identical-ten-rows') for name,values in predictions.items()}
        choice = actions(capabilities,profile,[],{'primary':bank},metric,['primary'],1)[0]
        result[metric.name] = dict(scores=scores,action=choice['action'],selected=choice['members'][0])
    return result


def package(path,role,metric):
    manifest=dict(schema_version=1,modality='tabular',target='label',metric=metric,
                  columns=[dict(name='signal',type='numeric')],splits={role:dict(format='csv',path=role+'.csv')})
    data=('signal,label\n'+''.join(f'{i},{int(i>=36)}\n' for i in range(48)) if role=='train'
          else 'signal\n41\n3\n46\n13\n')
    with zipfile.ZipFile(path,'w') as z:
        z.writestr('manifest.json',json.dumps(manifest))
        z.writestr(role+'.csv',data)
    return path


def main(root, selected_cases=None):
    root.mkdir(parents=True,exist_ok=False)
    save(root/'protocol.json',read(ROOT/'research/metric_interruption_1125/protocol.json'))
    result={}
    for case,alias,metric,expected,fits in [
            ('normal','macro-F1','macro_f1','completed',2),
            ('before_first_fit','AUC','roc_auc','partial',0),
            ('refit_failure','ROC-AUC','roc_auc','partial',1),
            ('hard_exit','macro-F1','macro_f1','partial',1),
            ('stream_cut','Accuracy','accuracy','completed',2)]:
        if selected_cases and case not in selected_cases:
            continue
        folder=root/case
        folder.mkdir()
        p,q=package(folder/'p.zip','train',alias),package(folder/'q.zip','test',alias)
        run=folder/'run';path=folder/'delivery'/'predictions.csv'
        key=folder/'dummy.fixture';key.write_text('test-only')
        env=dict(os.environ,SUPERAGENT_OFFLINE='1',AB_B1_OFFLINE='1',HF_HUB_OFFLINE='1',
                 OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',AB_B1_FIT_LOG=str(folder/'physical.jsonl'),
                 AB_B1_PER_WORKER_FIT_LIMIT='1',AIAA_RECOVERY_FIXTURE=case,AIAA_DUMMY_KEY=str(key),
                 PYTHONPATH=os.pathsep.join(map(str,(ROOT/'scripts/metric_interruption_observer',ROOT))))
        for k in ('SUPERAGENT_PLANNER_KEY_FILE','SUPERAGENT_PLANNER_LEDGER','SUPERAGENT_TASK_BUDGET_ID'):
            env.pop(k,None)
        def cli(tag,args):
            with (folder/(tag+'.log')).open('x') as log:
                subprocess.run([sys.executable,'-m','unified_agent',*map(str,args)],cwd=ROOT,env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,timeout=125,check=True)
        cli('solve',['solve',p,'--prediction-input',q,'--run-dir',run,'--output',path,'--fit-limit','2','--time-limit','80'])
        replay=folder/'again'/'predictions.csv'
        cli('predict',['predict',q,'--run-dir',run,'--output',replay])
        outcome=read(run/'agent'/'run_outcome.json');sel=read(run/'agent'/'final_selection.json')
        spec=read(run/'agent'/'task_spec.json');receipt=read(path.parent/'prediction_manifest.json')
        phy=folder/'physical.jsonl'
        workers=len(phy.read_text().splitlines()) if phy.is_file() else 0
        with path.open() as stream:
            rows=list(csv.DictReader(stream))
        assert len(rows)==4 and [r['row_id'] for r in rows]==list(map(str,range(4)))
        assert path.read_bytes()==replay.read_bytes() and spec['metric']==metric
        assert workers==outcome['attempted_classifier_fits']==fits and outcome['status']==receipt['delivery_status']==expected
        assert sel['primary_metric']==metric and outcome['new_api_requests']==int(case=='stream_cut')
        if metric=='roc_auc':
            assert receipt['score_class_order']==['0','1']
            assert all(abs(float(r['score_0'])+float(r['score_1'])-1)<1e-8 for r in rows)
        if case in ('hard_exit','refit_failure'):
            assert sel['model_format']=='verified_trial_v1' and not sel['refit']
        if case=='before_first_fit':
            assert sel['model_format']=='class_prior_v1'
        if case=='stream_cut':
            assert outcome['unresolved_api_requests']==1 and outcome['complete_run_course_cost_rmb'] is None
        result[case]=dict(status=expected,metric=metric,method=sel['family'],model_format=sel['model_format'],
            source_trial=sel.get('source_trial'),refit=sel['refit'],fits=workers,prediction_rows=len(rows),
            same_after_reload=True,prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            simulated_dispatches=outcome['new_api_requests'],unresolved_in_simulation=outcome['unresolved_api_requests'],
            live_api_requests=0)
        save(root/'results.json',result)
    save(root/'metric_preference.json',preference_probe())
    save(root/'completion.json',dict(status='complete',physical_fits=sum(r['fits'] for r in result.values()),live_api_requests=0))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    p.add_argument('--cases',nargs='+',choices=['normal','before_first_fit','refit_failure','hard_exit','stream_cut'])
    args=p.parse_args()
    main(args.output.resolve(),args.cases)
