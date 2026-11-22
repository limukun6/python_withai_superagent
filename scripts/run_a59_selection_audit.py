"""Paired group bootstrap on existing primary predictions; exploratory, no fitting/HTTP."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from scripts import a38_predict_guard
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.data import load_training_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run','training','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);manifest=read(a.run/'run_manifest.json')
    if sha(a.training)!=manifest['task_archive_sha256']:raise ValueError('Wrong training data')
    task=load_training_task(a.training);payload=joblib.load(a.run/'primary.joblib');split=read(a.run/'primary_split.json')
    records=read(a.run/'prediction_evidence.json');history=read(a.run/'history.json');values={}
    for candidate in ('word_char_C0.05','word_char_C0.1'):
        row=next(r for r in history if r['candidate']==candidate and r['split_id']=='primary')
        evidence=next(r for r in records if r['trial']==row['trial'])
        if (evidence['row_ids']!=split['search_indices'] or evidence['label_sha256']!=fingerprint(payload['y_search'])
                or evidence['prediction_sha256']!=fingerprint(evidence['predictions'])
                or evidence['evaluation_payload_sha256']!=sha(a.run/'primary.joblib')):raise ValueError('Evidence mismatch')
        values[candidate]=np.asarray(evidence['predictions'])
    y=np.asarray(payload['y_search']);delta=(values['word_char_C0.1']==y).astype(float)-(values['word_char_C0.05']==y).astype(float)
    groups=sample_groups(task,'shared_text')[split['search_indices']];unique,inverse=np.unique(groups,return_inverse=True)
    sums=np.bincount(inverse,weights=delta);sizes=np.bincount(inverse);rng=np.random.default_rng(181810);draw=[]
    for _ in range(2000):
        chosen=rng.integers(0,len(unique),size=len(unique));draw.append(sums[chosen].sum()/sizes[chosen].sum())
    low,high=np.quantile(draw,[.025,.975])
    result=dict(status='completed_exploratory',rows=len(y),text_groups=len(unique),C01_minus_C005_primary_accuracy=float(delta.mean()),
        paired_group_bootstrap_95_percentile_interval=[float(low),float(high)],interval_includes_zero=bool(low<=0<=high),
        draws=2000,seed=181810,actual_classifier_fits=0,real_api_requests=0,
        scope='Conditional fitted-model, post-selection primary-row uncertainty; not independent confirmatory inference or training variance',
        prediction_evidence_sha256=sha(a.run/'prediction_evidence.json'))
    save(a.output/'protocol.json',dict(batch='A59-read-only-selection-audit',runner_sha256=sha(Path(__file__)),
        original_manifest_sha256=sha(a.run/'run_manifest.json'),classifier_fitting_forbidden=True,network_forbidden=True))
    save(a.output/'completion.json',result);print(json.dumps(result))
