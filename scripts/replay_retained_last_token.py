"""Reload the retained research candidate with independently relocated frozen weights."""
import argparse
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np

from scripts.compare_regrouped_portfolios import lock
from scripts.unified_released_study import read, require, sha
from superagent.ab_complement import canonical_probabilities
from superagent.external_assets import activate_resource_directory
from superagent.local_sentence import describe, encode, installed
from superagent.models import predict_with_model
from superagent.prediction_evidence import fingerprint


def run(root):
    folder=root/'last-token-subset'
    agent=folder/'expanded/run/agent'
    family=read(folder/'results.json')['additional_member']
    row=next(r for r in read(agent/'history.json') if r['candidate']==family and r['split_id']=='primary')
    worker=agent/f"trial_{row['trial']}.joblib"
    item=joblib.load(worker)
    payload=joblib.load(agent/'primary.joblib')
    receipt=read(root/'last-token-installed.json')
    source=next(v for v in (root/'last-token-library').iterdir() if (v/'modules.json').is_file() and describe(v)['signature'][:20] in family)
    os.environ.pop('SUPERAGENT_PAIRED_FEATURE_CACHE',None)
    with TemporaryDirectory() as temp:
        opaque=Path(temp)/'asset'
        for relative in describe(source)['files']:
            target=opaque/relative
            target.parent.mkdir(parents=True,exist_ok=True)
            os.link(source/relative,target)
        activate_resource_directory(Path(temp))
        signature=describe(opaque)['signature']
        require(len(receipt)==1 and any(s['signature']==signature for s in installed()),'Relocated encoder missing')
        rows=payload['X_search']
        model=item['model']
        predictions=predict_with_model(payload['kind'],model,rows)
        _,probabilities=canonical_probabilities(model,rows)
        old=np.asarray(item['search_scores'])
        require(list(item['score_classes'])==[0,1], 'Class order changed')
        fixes=int(np.sum(predictions!=item['search_predictions']))
        error=float(np.max(np.abs(old-probabilities)))
        # Verify a declared alternative padding side without modifying any original hardlinked bytes.
        sample=['A compact sentence.', 'Two nearby cities are linked by a new train line.', '']
        regular=encode(describe(opaque),sample)
        config=opaque/'tokenizer_config.json'
        edited=json.loads(config.read_text())
        edited['padding_side']='left'
        config.unlink()
        config.write_text(json.dumps(edited))
        left=describe(opaque)
        activate_resource_directory(Path(temp))
        padded=encode(left,sample)
        cosines=(regular*padded).sum(axis=1)/(np.linalg.norm(regular,axis=1)*np.linalg.norm(padded,axis=1))
        require(fixes==0 and error<=.01 and cosines.min()>=.999,'Retained classifier or mixed-length embeddings changed beyond tolerance')
        record=dict(source_classifier=sha(worker),rows=len(rows),original_predictions=fingerprint(item['search_predictions']),
                    replayed_predictions=fingerprint(predictions),changed_labels=fixes,max_probability_difference=error,
                    explicit_left_vs_default_min_cosine=float(cosines.min()),explicit_left_has_distinct_cache_identity=left['signature']!=signature,
                    singleton_or_batch_probe_count=len(sample),new_supervised_fits=0,model_api_requests=0,course_validation_opened=False)
    lock(root/'last-token-retained-reload.json',record)
    print(record,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=lambda s:Path(s).resolve(),required=True)
    run(parser.parse_args().root)
