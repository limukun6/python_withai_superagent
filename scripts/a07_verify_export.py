"""Independent-process frozen model/prediction verification after run completion."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import joblib
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-root',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--holdout',required=True);p.add_argument('--predictions',required=True);p.add_argument('--output',required=True)
    p.add_argument('--skip-score',action='store_true',help='Recheck delivery only; do not reevaluate held labels')
    a=p.parse_args();sys.path.insert(0,str(Path(a.source_root).resolve()))
    from superagent.metrics import MetricSpec
    from superagent.models import predict_with_model,probability_with_model
    root=Path(a.run_dir);selection=json.loads((root/'final_selection.json').read_text(encoding='utf-8'))
    model_path=root/selection['model_file'];digest=hashlib.sha256(model_path.read_bytes()).hexdigest()
    if digest!=selection['model_sha256']:raise ValueError('Frozen model hash changed')
    bundle=joblib.load(model_path);held=joblib.load(a.holdout);model=bundle['model'];kind=bundle['kind']
    prediction=predict_with_model(kind,model,held['X'])
    with open(a.predictions,encoding='utf-8',newline='') as f:rows=list(csv.DictReader(f))
    if len(rows)!=len(prediction):raise ValueError('Prediction row count mismatch')
    if [int(r['row_id']) for r in rows]!=list(range(len(rows))):raise ValueError('Prediction row order mismatch')
    if [r['prediction'] for r in rows]!=[str(v) for v in prediction.tolist()]:raise ValueError('Prediction values differ across processes')
    spec=json.loads((root/'task_spec.json').read_text(encoding='utf-8'));metric=MetricSpec.from_payload({'metric_spec':spec['metric_spec']})
    result=None if a.skip_score else metric.evaluate_model(kind,model,held['X'],held['y'],prediction)
    record=dict(reload_identical=True,rows=len(rows),primary_metric=metric.name,score=result['score'] if result else None,accuracy=result['accuracy'] if result else None,
                heldout_scored=not a.skip_score,
                model_sha256=digest,original_row_ids_sha256=hashlib.sha256(json.dumps(held['row_ids']).encode()).hexdigest(),
                selected_candidate=selection['family'],scope='Outer development holdout after model freeze; not official test')
    if metric.requires_scores:
        classes,scores=probability_with_model(kind,model,held['X'])
        manifest=json.loads((Path(a.predictions).parent/'prediction_manifest.json').read_text(encoding='utf-8'))
        if manifest['score_class_order']!=classes.tolist():raise ValueError('Score class order mismatch')
        actual=np.array([[float(r[f'score_{i}']) for i in range(len(classes))] for r in rows])
        if not np.allclose(actual,scores,rtol=0,atol=1e-12):raise ValueError('Export probabilities differ')
        record['probabilities_identical']=True
    Path(a.output).write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8');print(json.dumps(record))


if __name__=='__main__':main()
