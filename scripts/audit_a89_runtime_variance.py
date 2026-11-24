"""Post-hoc read-only comparison of an identical primary action's runtime/identity, no fits/API."""
import argparse
import json
from pathlib import Path
import joblib
from scripts import a38_predict_guard
from scripts.a_batch_common import save
from superagent.scheduler_bridge import fingerprint

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--before',type=Path,required=True)
    p.add_argument('--after',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    candidate='supported_field_interactions';records=[];parameters=[]
    for root in (a.before,a.after):
        history=json.loads((root/'history.json').read_text());row=next(r for r in history if r['candidate']==candidate and r['split_id']=='primary')
        evidence=next(r for r in json.loads((root/'prediction_evidence.json').read_text()) if r['trial']==row['trial'])
        model=joblib.load(root/f"trial_{row['trial']}.joblib")['model'];params=model.classifier.get_all_params();parameters.append(params)
        records.append(dict(run=str(root),trial=row['trial'],candidate=candidate,worker_wall_seconds=row['worker_wall_seconds'],
            training_seconds=row['training_seconds'],declared_parameters=row['params'],fitted_parameters_sha256=fingerprint(params),
            prediction_sha256=evidence['prediction_sha256'],evaluation_payload_sha256=evidence['evaluation_payload_sha256'],
            label_sha256=evidence['label_sha256'],primary_score=row['search_score']))
    changed={key:[parameters[0].get(key),parameters[1].get(key)] for key in set(parameters[0])|set(parameters[1]) if parameters[0].get(key)!=parameters[1].get(key)}
    save(a.output,dict(post_hoc=True,records=records,fitted_parameter_differences=changed,
        matching_fields={key:records[0][key]==records[1][key] for key in ['declared_parameters','prediction_sha256','evaluation_payload_sha256','label_sha256','primary_score']},
        runtime_cause_not_identified=True,old_reused_L1_runtime_not_fresh_efficiency_evidence=True,
        actual_classifier_fits=0,real_api_requests=0))
    print(dict(runtime_ratio=records[1]['training_seconds']/records[0]['training_seconds'],fitted_parameter_differences=list(changed),
               prediction_same=records[0]['prediction_sha256']==records[1]['prediction_sha256']))
