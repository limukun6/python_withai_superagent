"""Check current binary-AUC scoring invariance with named normalized probabilities; no fitting/API."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.run_sa5_semantic_panel import save,sha
from superagent.metrics import MetricSpec


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1]
    save(a.output/'protocol.json',dict(batch='SA18-binary-auc-orientation',runner_sha256=sha(Path(__file__)),
        metric_source_sha256=sha(root/'superagent/metrics.py'),
        plan_sha256=sha(root/'docs/batches/SA18_RECOVERY_REALIGNMENT_20261127_zh.md'),
        maximum_classifier_fits=0,maximum_real_API_requests=0,source='fixed synthetic score matrices, no trained classifier'))
    y=np.array([0,1,0,1,1,0]);examples=[]
    for p1 in [np.array([.1,.8,.3,.4,.6,.9]),np.array([.5]*6),np.array([0,1,0,1,1,0]),np.array([1,0,1,0,0,1])]:
        probabilities=np.column_stack([1-p1,p1]);prediction=np.argmax(probabilities,axis=1)
        a_score=MetricSpec('roc_auc',(0,1)).evaluate(y,prediction,probabilities,[0,1])['score']
        b_score=MetricSpec('roc_auc',(1,0)).evaluate(y,prediction,probabilities,[0,1])['score']
        examples.append(dict(class1_auc=a_score,class0_auc=b_score,equal=bool(np.isclose(a_score,b_score))))
    result=dict(status='passed' if all(r['equal'] for r in examples) else 'failed',examples=examples,
        actual_classifier_fits=0,real_API_requests=0,course_cost_rmb=0,
        boundary='internal binary AUC with normalized named class probabilities; an unnamed positive output column remains a separate output-contract question')
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result));return result['status']!='passed'


if __name__=='__main__':raise SystemExit(main())
