"""Twelve objective traces: controlled predictions, real selection/finalization code."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import joblib
from a_batch_common import Journal


class FixedPredictor:
    def __init__(self, minority):self.minority=minority;self.classes_=np.array([0,1])
    def predict(self,x):return (np.asarray(x)[:,0]>=14).astype(int) if self.minority else np.zeros(len(x),dtype=int)
    def predict_proba(self,x):
        values=np.where(self.predict(x),.9,.1)
        return np.column_stack((1-values,values))


def worker(payload_path,candidate,destination,timeout):
    from superagent.metrics import MetricSpec
    from superagent.prediction_evidence import make_prediction_record
    p=joblib.load(payload_path);metric=MetricSpec.from_payload(p);model=FixedPredictor(candidate.endswith('C1.0'))
    metrics={'classifier_fits':0,'training_seconds':0.,'metric_spec':metric.record()}
    record={'model':model,'metrics':metrics}
    if p.get('mode')!='refit':
        pred=model.predict(p['X_search']);a=metric.evaluate_model(p['kind'],model,p['X_search'],p['y_search'],pred)
        b=metric.evaluate_model(p['kind'],model,p['X_train'],p['y_train'])
        metrics.update(primary_metric=metric.name,search_score=a['score'],train_score=b['score'],
                       search_accuracy=a['accuracy'],train_accuracy=b['accuracy'],score_gap=b['score']-a['score'],generalization_gap=b['accuracy']-a['accuracy'])
        record.update(search_predictions=pred,prediction_record=make_prediction_record(pred,p['y_search'],hashlib.sha256(Path(payload_path).read_bytes()).hexdigest()))
        if metric.requires_scores:record.update(search_scores=model.predict_proba(p['X_search']),score_classes=model.classes_)
    joblib.dump(record,destination)
    return model,{'status':'completed',**metrics}


class FixedOrder:
    def propose(self,state):
        from superagent.contracts import ExperimentProposal
        r=state.record();suffix='C0.1' if not r['history'] else 'C1.0'
        c=next(c for c in r['available_candidates'] if c['id']=='numeric_tabular_logistic_'+suffix)
        return ExperimentProposal.from_decision(r['next_trial'],c,'Predeclared fixed prediction probe',None,r['history'])


def legal(name,root):
    from superagent.benchmark import run_one
    from superagent.data import DevelopmentTask
    # Repeated numeric values give both inner portions identical controlled class counts.
    x=np.tile(np.arange(20),(3,1)).reshape(-1,1);y=np.tile([0]*18+[1]*2,3)
    task=DevelopmentTask('numeric_tabular','anonymous_probe',x,y,x[:20],y[:20],{'metric':name})
    def split(task,seed,mode):return np.arange(40),np.arange(40,60),{'scope':'controlled diagnostic indices'}
    with patch('superagent.benchmark.train_candidate',side_effect=worker),patch('superagent.benchmark.grouped_partition',side_effect=split),patch('superagent.confirmation.grouped_partition',side_effect=split):
        summary=run_one(task,'local_interleaved_routes',707,root,None,max_trials=2,time_limit=60,trial_timeout=10,
                        confirmation_repeats=3,confirmation_top_k=2,refit_full_train=True,decision_policy=FixedOrder())
    names=['task_contract.json','decision_states.json','experiment_results.json','confirmation.json','final_selection.json','task_spec.json']
    records={n:json.loads((root/n).read_text(encoding='utf-8')) for n in names}
    expected='numeric_tabular_logistic_'+('C0.1' if name=='accuracy' else 'C1.0')
    checks={'selected':summary['selected_candidate']==expected,'refit':summary['refit_succeeded'],
            'task':records['task_contract.json']['metric']['name']==name,
            'states':all(r['task_spec']['metric']['name']==name for r in records['decision_states.json']),
            'results':all(r['measurements']['primary_metric']==name for r in records['experiment_results.json']),
            'confirmation':records['confirmation.json']['primary_metric']==name,
            'export':records['task_spec.json']['metric_spec']['name']==name and records['final_selection.json']['primary_metric']==name}
    return {'metric':name,'case':'legal','passed':all(checks.values()),'checks':checks,'selected':summary['selected_candidate'],
            'expected':expected,'observed_score':summary['official_validation_score'],'simulated_worker_calls':7,'classifier_fits':0}


def boundary(name,mode):
    from superagent.contracts import ExperimentProposal,ExperimentResult
    from superagent.metrics import MetricSpec
    expected=MetricSpec(name,(0,1));config={'id':'candidate','params':{}}
    proposal=ExperimentProposal.from_decision(1,config,'Predeclared boundary probe',None,[])
    row={'trial':1,'candidate':'candidate','params':{},'status':'completed','primary_metric':name,'search_score':.7,'search_accuracy':.8}
    if mode=='conflict':
        actual=expected.record()
        if name in ('binary_f1','roc_auc'):actual['labels']=[1,0]
        else:actual['name']='macro_f1' if name=='accuracy' else 'accuracy'
        row['metric_spec']=actual
    try:
        result=ExperimentResult.from_trial(row,proposal,expected).record()
        status=result.get('metric_identity',{}).get('status','confirmed')
        passed=mode=='missing' and status=='unknown'
        return {'metric':name,'case':mode,'passed':passed,'status':status,'record':result}
    except ValueError as e:return {'metric':name,'case':mode,'passed':mode=='conflict','status':'rejected','error':str(e)}


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--output',required=True);p.add_argument('--validate-only',action='store_true')
    a=p.parse_args();sys.path.insert(0,str(Path(a.source_root).resolve()))
    j=Journal(a.output,a.source_root,{'batch':'A-06','metrics':['accuracy','macro_f1','binary_f1','roc_auc'],
        'cases':['legal','conflict','missing'],'classifier_fit_limit':0,'simulated_worker_call_limit':28,'validate_only':a.validate_only})
    results=[]
    if not a.validate_only:
        for name in ('accuracy','macro_f1','binary_f1','roc_auc'):
            try:results.append(legal(name,j.out/name))
            except Exception as e:results.append({'metric':name,'case':'legal','passed':False,'error':repr(e)})
            results.extend(boundary(name,mode) for mode in ('conflict','missing'))
    j.finish(results,cases=len(results),passed=sum(r['passed'] for r in results),classifier_fits=0,
             simulated_worker_calls=sum(r.get('simulated_worker_calls',0) for r in results))
    print(json.dumps(results,ensure_ascii=False))


if __name__=='__main__':main()
