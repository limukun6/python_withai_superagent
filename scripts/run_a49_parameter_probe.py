"""Five frozen real models, bounded parameter observations and artificial wrong declarations."""
import argparse
import json
from pathlib import Path

import joblib

from scripts import a38_predict_guard
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.ab_run import sources
from superagent.estimator_facts import verify_configuration_claim
from tools.a49_parameter_observer import observe,compare


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);frozen=sources();results=[]
    save(root/'protocol.json',dict(batch='A49-fitted-parameter-prototype',source_files=frozen,
        runner_sha256=sha(Path(__file__)),prototype_sha256=sha(Path(__file__).resolve().parents[1]/'tools/a49_parameter_observer.py'),
        classifier_fits_forbidden=True,network_forbidden=True,production_modified=False,external_labels_opened=False))
    for name in ('coupon','preference','sms','photo','bean'):
        directory=args.models/(name+'_complete');selected=read(directory/'final_selection.json')
        path=directory/selected['model_file']
        if sha(path)!=selected['model_sha256']:raise ValueError('Frozen model changed')
        model=joblib.load(path)['model'];facts=observe(model)
        original=next(r for r in read(directory/'history.json') if r['candidate']==selected['family'])
        checks=compare(original['params'],facts)
        eligible=[c for c in checks if c['status']=='confirmed' and type(c['observed']) in (float,int)]
        fault=None
        if eligible:
            chosen=eligible[0];bad={**original['params'],chosen['parameter']:chosen['observed']*10+1}
            badchecks=compare(bad,facts)
            old=verify_configuration_claim(dict(configuration_claim={'classifier':facts['family']}),facts)
            if old['status']!='confirmed' or not any(c['status']=='contradicted' for c in badchecks):raise ValueError('Artificial wrong parameter was not caught')
            fault=dict(parameter=chosen['parameter'],old_family_only_check=old['status'],prototype_check='contradicted',
                fault_injected=True,actual_model_modified=False)
        results.append(dict(task=name,model_sha256=sha(path),observed=facts,actual_declared_parameter_checks=checks,
            wrong_declaration_probe=fault,probe_status='passed' if fault else 'unsupported_no_numeric_declared_parameter',
            deployed_in_agent=False))
    save(root/'results.json',results)
    save(root/'completion.json',dict(status='completed',models=len(results),injected_wrong_declarations_caught=sum(r['wrong_declaration_probe'] is not None for r in results),
        natural_contradictions=sum(c['status']=='contradicted' for r in results for c in r['actual_declared_parameter_checks']),
        actual_classifier_fits=0,real_api_requests=0,source_unchanged=sources()==frozen,production_integration_performed=False))
    print(json.dumps(read(root/'completion.json')))
