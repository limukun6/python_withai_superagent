"""Real training with a deterministic simulated planner; no API requests."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['baseline','fixed'],required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    source=lambda:{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((REPO/'superagent').glob('*.py'))}
    frozen=source();save('protocol.json',dict(phase=args.phase,source_sha256=frozen,
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),api_requests=0,
        seed=707,planned_fits=6,scope='Controlled fault injection with deterministic simulated planner; not LLM performance'))
    start=time.monotonic();results=[];fit_count=0
    try:
        import numpy as np
        from superagent.benchmark import run_one
        from superagent.checked_history_policy import CheckedHistoryPolicy,decision_context
        from superagent.data import DevelopmentTask
        from superagent.contracts import DecisionState
        from superagent import targeted_feedback
        real_facts=targeted_feedback.candidate_facts
        target='numeric_tabular_extra_trees_leaf1'
        def incomplete_catalog(kind,candidate):
            if candidate['id']==target:
                return {'classifier':None,'source':'injected_unknown','reason':'controlled missing metadata'}
            return real_facts(kind,candidate)
        rng=np.random.default_rng(133)
        x=rng.normal(size=(180,5));y=(x[:,0]+.4*x[:,1]>0).astype(int)
        task=DevelopmentTask('numeric_tabular','controlled_fixture',x,y,None,None,{'metric':'accuracy'})
        class SimulatedClient:
            def __init__(self):self.contexts=[]
            def ask(self,context,logger,role):
                self.contexts.append(deepcopy(context));first=len(self.contexts)==1
                candidate=target if first else 'numeric_tabular_logistic_C1.0'
                conflict=any((r.get('configuration_fact_check') or {}).get('status')=='contradicted'
                             for r in context.get('checks',[]))
                return dict(candidate=candidate,task_kind='numeric_tabular',configuration_claim={'classifier':'logistic'},
                    reason='Saw fitted-classifier conflict; revisit a known configuration.' if conflict else 'No explicit conflict was supplied.',
                    review='Controlled planner response, not an independent reasoning result.',
                    hypothesis='Trial 1 search accuracy was 99.99%.',evidence_trials=[1],
                    expected_change=dict(metric='search_score',reference_trial=1,direction='increase',min_delta=.99),slice_expectation=None)
        for checked in (False,True):
            name='checked' if checked else 'raw';client=SimulatedClient()
            with patch('superagent.targeted_feedback.candidate_facts',side_effect=incomplete_catalog),patch(
                'superagent.checked_history_policy.candidate_facts',side_effect=incomplete_catalog):
                summary=run_one(task,'local_interleaved_routes',707,out/name,None,
                    max_trials=3,time_limit=180,trial_timeout=60,portfolio='expanded',anchor_first=True,
                    confirmation_repeats=1,refit_full_train=False,
                    decision_policy=CheckedHistoryPolicy(client,None,checked))
            fit_count+=summary['classifier_fit_accounting']['completed_classifier_fits']
            save(name+'_contexts.json',client.contexts)
            trials=json.loads((out/name/'trials.json').read_text(encoding='utf-8'))
            states=json.loads((out/name/'decision_states.json').read_text(encoding='utf-8'))
            second=trials[1];context=client.contexts[1]
            results.append(dict(arm=name,observed_family=second.get('estimator',{}).get('family'),
                recorded_conflict=second.get('configuration_fact_check',{}).get('status')=='contradicted',
                state_has_observation='estimator' in states[2]['history'][1],
                audit_ran_without_assessment=bool(second.get('explanation_audit')),
                hypothesis_audited=any(w.get('field')=='hypothesis' for w in (second.get('explanation_audit') or {}).get('warnings',[])),
                next_has_conflict=any((r.get('configuration_fact_check') or {}).get('status')=='contradicted' for r in context.get('checks',[])),
                next_has_failed_expectation=any((r.get('hypothesis_check') or {}).get('status')=='not_supported' for r in context.get('checks',[])),
                raw_leak=any(k in h for h in context['history'] for k in ('configuration_fact_check','explanation_audit','hypothesis_check')),
                corrected_response='Saw fitted-classifier conflict' in trials[2]['proposal']['reason'],
                completed_trials=summary['trials_attempted'],fallbacks=summary['fallback_count'],simulated_calls=len(client.contexts)))
            save('results.json',results)
        raw,checked=results
        checks=[all(r['recorded_conflict'] for r in results),all(r['state_has_observation'] for r in results),
            all(r['audit_ran_without_assessment'] for r in results),all(r['hypothesis_audited'] for r in results),
            checked['next_has_conflict'],checked['next_has_failed_expectation'],
            not raw['next_has_conflict'] and not raw['raw_leak'],checked['corrected_response'] and not raw['corrected_response'],
            all(r['completed_trials']==3 and r['fallbacks']==0 and r['simulated_calls']==2 for r in results)]
        unchanged=frozen==source()
        completion=dict(status='complete',phase=args.phase,passed=sum(checks),cases=len(checks),
            checks=checks,api_requests=0,simulated_calls=4,classifier_fits=fit_count,
            elapsed_seconds=time.monotonic()-start,source_unchanged=unchanged)
        save('completion.json',completion);print(json.dumps(completion))
        return int(not unchanged or (args.phase=='fixed' and not all(checks)))
    except Exception as error:
        save('completion.json',dict(status='failed',error=f'{type(error).__name__}: {error}',
            completed_fits_from_finished_arms=fit_count,api_requests=0,elapsed_seconds=time.monotonic()-start))
        raise

if __name__=='__main__':raise SystemExit(main())
