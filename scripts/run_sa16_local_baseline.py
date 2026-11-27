"""Seal the pre-auto local-success/output-missing counterexamples, zero fit/API."""
import argparse
import json
from pathlib import Path
import socket
import tempfile
from unittest.mock import patch

from scripts.run_sa5_semantic_panel import save,sha
from superagent.semantic_contract import resolve_training_task
from tests.test_auto_semantics import standard_case
from tests.test_semantic_contract import Client


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
    save(a.output/'protocol.json',dict(batch='SA16-pre-auto-baseline',base='b620b83',runner_sha256=sha(Path(__file__)),
        reader_sha256=sha(root/'superagent/semantic_contract.py'),maximum_classifier_fits=0,maximum_real_API_requests=0,
        scope='developer mechanism counterexamples; no trained model or real LLM answer'))
    rows=[]
    with tempfile.TemporaryDirectory() as temp,patch.object(socket,'create_connection',side_effect=AssertionError('No network')):
        for manifest in [False,True]:
            path,answer,expected=standard_case(Path(temp)/f'{manifest}.zip',manifest);client=Client([answer])
            task=resolve_training_task(path,client,prediction_package=path)
            rows.append(dict(case='explicit_manifest' if manifest else 'standard_csv',input_accepted=True,training_rows=len(task.y_train),
                             complete_output_contract=task.spec.get('output_recipe')==expected,simulated_model_requests=client.request_count))
    result=dict(status='baseline_sealed',cases=rows,actual_classifier_fits=0,real_API_requests=0,course_cost_rmb=0,
                complete_output_contracts=sum(r['complete_output_contract'] for r in rows))
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result))


if __name__=='__main__':main()
