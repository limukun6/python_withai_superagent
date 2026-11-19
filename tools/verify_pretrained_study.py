"""Audit the paired report using frozen artifacts. Reject any new encoder forward."""
import argparse
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np

from .expanded_feedback_pilot import sha
from .pretrained_study import SEED, read, train_only, write
from .search_budget_study import outer_partition
from superagent.models import predict_with_model
from superagent.pretrained_choice import Encoder, FILES, digest


def verify(root, archives):
    protocol = read(root/'protocol.json')
    results = read(root/'results.json')
    repo = Path(__file__).resolve().parents[1]
    if any(sha(repo / path) != frozen for path,frozen in protocol['source'].items()):
        raise ValueError('Source differs from frozen protocol')
    checks = {}
    for name, archive in archives.items():
        accessed=[]
        if sha(archive)!=protocol['archives'][name]:
            raise ValueError('Wrong input archive')
        with train_only(archive,accessed):
            task,x,y,split = outer_partition(archive, SEED)
        paired = read(root/name/'outer_predictions.json')
        if split != protocol['partitions'][name] or split['held_rows']!=paired['rows'] or y.tolist()!=paired['truth']:
            raise ValueError('Frozen outer split mismatch')
        bases = {}
        for arm in ('control','choice'):
            dest = root/name/arm
            summary = read(dest/'summary.json')
            account = results[name]['arms'][arm]
            if sha(dest/'selected_model.joblib')!=account['model_sha256']:
                raise ValueError('Changed model')
            model = joblib.load(dest/'selected_model.joblib')
            with patch.object(Encoder, 'load', side_effect=AssertionError('Verification cannot fetch or load a model')), \
                 patch.object(Encoder, 'forward', side_effect=AssertionError('Verification cannot spend tokens')):
                values = predict_with_model(task.kind, model['model'], x)
            if values.tolist()!=paired[arm] or int((values==y).sum())!=account['correct']:
                raise ValueError('Incorrect outer predictions')
            bases[arm]=values
            if not summary['confirmation']['valid_confirmation'] or not summary['refit_succeeded']:
                raise ValueError('Confirmation or refit did not finish')
        before,after=bases['control'],bases['choice']
        if int(((after==y)&(before!=y)).sum())!=results[name]['corrected'] or int(((before==y)&(after!=y)).sum())!=results[name]['lost']:
            raise ValueError('Paired differences disagree')
        checks[name] = dict(accessed=accessed,rows=len(y), prediction_replay='exact',classifier_fits=0,model_forwards=0)
    for loadfile in root.glob('*/*/model_loads.jsonl'):
        location = Path(read(loadfile.parent/'pretrained_runtime.json')['weights'])
        for name,(_,frozen,size) in FILES.items():
            path=location/name
            if path.stat().st_size!=size or digest(path)!=frozen:
                raise ValueError('Model asset verification failed')
        checks['official_weights']='all three downloaded files match the pinned SHA-256 and size'
        break
    write(root/'verification.json',checks)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('task1','task2','output'):
        p.add_argument('--'+arg,required=True,type=Path)
    a=p.parse_args()
    verify(a.output.resolve(),{'task1':a.task1,'task2':a.task2})
