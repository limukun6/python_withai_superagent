"""Seal deterministic fixture decisions, validate tamper rejection, then four real fits."""
import argparse
from copy import deepcopy
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_a2_state_study import read,save,sha,sources
from superagent.contracts import ExperimentProposal
from tools.a2_action_evaluation import evaluate,validate_inputs


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    prepared=Path(a.prepared);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);calls=out/'fixture_calls';calls.mkdir()
    states={i:read(prepared/f'dry_bean_prefix{i}.json') for i in (1,2)}
    common=[c for c in states[1]['available_candidates'] if c in states[2]['available_candidates']][:2]
    if len(common)!=2:raise ValueError('Smoke needs two shared executable actions')
    decisions=[]
    for prefix in (1,2):
        state=states[prefix]
        for checked in (False,True):
            config=common[1] if prefix==2 and checked else common[0]
            proposal=ExperimentProposal.from_decision(state['next_trial'],config,'Predeclared evaluator fixture',None,state['history'])
            decisions.append(dict(task='dry_bean',prefix=prefix,checked=checked,proposal=proposal.record()))
    binding=dict(fixture=True,source_sha256=sources(),prepared_protocol_sha256=sha(prepared/'protocol.json'))
    save(calls/'protocol.json',binding);save(calls/'results.json',decisions);save(calls/'completion.json',dict(status='complete',valid_proposals=4,api_requests=0))
    checks=[]
    mutations={'duplicate':decisions+[decisions[0]],'missing':decisions[:-1],
               'unknown_arm':[{**decisions[0],'checked':'true'},*decisions[1:]],
               'unavailable_candidate':deepcopy(decisions),'changed_params':deepcopy(decisions)}
    mutations['unavailable_candidate'][0]['proposal']['candidate']='missing'
    mutations['changed_params'][0]['proposal']['params']['unexpected']=True
    for name,values in mutations.items():
        directory=out/name;directory.mkdir();save(directory/'protocol.json',binding)
        save(directory/'completion.json',dict(status='complete'));save(directory/'results.json',values)
        try:validate_inputs(prepared,directory,True)
        except ValueError as error:checks.append(dict(case=name,rejected=True,error=str(error)))
        else:raise AssertionError(f'Invalid fixture admitted: {name}')
    try:validate_inputs(prepared,calls,False)
    except ValueError:checks.append(dict(case='fixture_as_real',rejected=True))
    else:raise AssertionError('Fixture accepted as real')
    save(out/'input_checks.json',checks)
    evaluate(prepared,calls,out/'validate_only',fixture=True,validate_only=True)
    evaluate(prepared,calls,out/'evaluation',fixture=True)
    done=read(out/'evaluation/completion.json')
    if done['completed_classifier_fits']!=4 or done['reused_evaluations']!=4:raise AssertionError('Unexpected deduplication counts')
    save(out/'completion.json',dict(status='complete',fixture=True,preflight_checks=len(checks),
        classifier_fits=4,api_requests=0,evaluator=done,scope='Evaluator verification, not model feedback effectiveness'))
    print(done)


if __name__=='__main__':main()
