"""Run the frozen production release on the same anonymous SMS files and numerical caps."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys

from sklearn.metrics import f1_score

from scripts.run_local_sentence_transfer import samples
from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT,env,read,require,save,sha


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('source','previous','weights','root'):
        p.add_argument('--'+k,type=lambda s:Path(s).resolve(),required=True)
    a=p.parse_args()
    study=a.root/'discovered-single-text'
    old=a.root/'discovered-text-control'
    old.mkdir(exist_ok=False)
    recorded=read(study/'lock.json')
    source_hash=source_identity(a.previous)
    archived=read(a.root/'anonymous-choice/anonymous_default/protocol.json')
    require(source_hash==archived['runtime_sha256'],'Control does not match the frozen release')
    task,features=study/'dataset.zip',study/'features.zip'
    require(sha(task)==recorded['task'] and sha(features)==recorded['features'],'Training inputs changed')
    save(old/'lock.json',dict(prereg=sha(ROOT/'research/paired_encoder_search_1127/discovery_text_control_protocol.json'),
                             source=source_hash,inputs=sha(study/'lock.json'),previously_observed=sha(study/'autonomous.json')))
    log=old/'physical.jsonl'
    environment=env(a.weights,log)
    for k in tuple(environment):
        if k.startswith('SUPERAGENT_') and (k.endswith('_WEIGHTS') or k in ('SUPERAGENT_PAIRED_FEATURE_CACHE','SUPERAGENT_WEIGHTS_DIR','SUPERAGENT_PLANNER_KEY_FILE')):
            environment.pop(k)
    environment['PYTHONPATH']=str(a.previous/'scripts/ab_acceptance_guard')
    run=old/'run'
    destination=old/'predictions.csv'
    replay=old/'reloaded.csv'
    for name,args in [('solve',['solve',task,'--fit-limit','18','--time-limit','1200','--weights-dir',a.weights,
                                '--prediction-input',features,'--run-dir',run,'--output',destination]),
                      ('predict',['predict',features,'--weights-dir',a.weights,'--run-dir',run,'--output',replay])]:
        with (old/(name+'.log')).open('x') as stream:
            subprocess.run([sys.executable,'-m','unified_agent',*map(str,args)],cwd=a.previous,env=environment,
                           stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,check=True,timeout=1350)
    require(sha(destination)==sha(replay),'Reload changed control predictions')
    def output(path):
        with path.open(newline='') as f:
            rows=list(csv.DictReader(f))
        require([int(r['row_id']) for r in rows]==list(range(160)),'Prediction order changed')
        return [r['prediction'] for r in rows]
    before=output(replay)
    after=output(study/'predictions.csv')
    expected=[y for _,y in samples(a.source)[1]]
    state=read(run/'agent/run_outcome.json')
    fits=len(log.read_text().splitlines())
    require(state['status']=='completed' and state['attempted_classifier_fits']==fits<=18 and state['new_api_requests']==0,'Incomplete control')
    result=dict(rows=len(before),correct=sum(x==y for x,y in zip(before,expected)),
        macro_f1=f1_score(expected,before,average='macro'),expanded_macro_f1=f1_score(expected,after,average='macro'),
        gains=sum(o!=t and n==t for o,n,t in zip(before,after,expected)),losses=sum(o==t and n!=t for o,n,t in zip(before,after,expected)),
        selected=read(run/'agent/decisions.json')[-1]['choice']['members'],fits=fits,independent_reload=True,
        prediction_sha256=sha(replay),model_api_requests=0,comparator_registered_after_expanded_result=True)
    save(old/'result.json',result)
    print(result,flush=True)
