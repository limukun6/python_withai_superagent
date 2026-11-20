"""Extra zero-fit checks on sealed state/context identity, dependencies and cost plan."""
import argparse
from copy import deepcopy
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_a2_state_study import read,save,sha
from tools.a2_action_evaluation import validate_inputs,evaluate


def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--calls',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False);results=[]
    cases=['state_bytes','context_hash','unregistered_seeds','dependency','calls_binding','unfinished_calls','nonboolean_arm']
    for case in cases:
        folder=out/case;folder.mkdir();prepared=folder/'prepared';prepared.mkdir();calls=folder/'calls';calls.mkdir()
        for f in Path(a.prepared).glob('*.json'):shutil.copyfile(f,prepared/f.name)
        for f in Path(a.calls).glob('*.json'):shutil.copyfile(f,calls/f.name)
        entries=read(prepared/'states.json');entry=next(e for e in entries if e['task']=='dry_bean' and e['prefix']==1)
        if case=='state_bytes':
            state=read(prepared/entry['file']);state['next_trial']=999;save(prepared/entry['file'],state)
        if case=='context_hash':entry['raw_context_sha256']='0'*64;save(prepared/'states.json',entries)
        if case in ('unregistered_seeds','dependency'):
            protocol=read(prepared/'protocol.json')
            if case=='unregistered_seeds':protocol['evaluation_seeds']=[707,2707]
            else:protocol['dependencies']['numpy']='0.0.invalid'
            save(prepared/'protocol.json',protocol)
            binding=read(calls/'protocol.json');binding['prepared_protocol_sha256']=sha(prepared/'protocol.json');save(calls/'protocol.json',binding)
        if case=='calls_binding':
            binding=read(calls/'protocol.json');binding['prepared_protocol_sha256']='0'*64;save(calls/'protocol.json',binding)
        if case=='unfinished_calls':save(calls/'completion.json',{'status':'stopped'})
        if case=='nonboolean_arm':
            rows=read(calls/'results.json');rows[0]['checked']=1;save(calls/'results.json',rows)
        try:validate_inputs(prepared,calls,True)
        except ValueError as error:results.append({'case':case,'rejected':True,'error':str(error)})
        else:raise AssertionError(f'Bad input accepted: {case}')
    evaluate(a.prepared,a.calls,out/'final_validate_only',fixture=True,validate_only=True)
    save(out/'results.json',results);save(out/'completion.json',dict(status='complete',cases=len(results),passed=len(results),classifier_fits=0,api_requests=0))
    print({'passed':len(results),'classifier_fits':0,'api_requests':0})


if __name__=='__main__':main()
