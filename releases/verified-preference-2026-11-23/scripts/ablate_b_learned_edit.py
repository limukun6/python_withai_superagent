"""Mask the learned edit block on each already held internal partition. No score selects a model."""
import argparse
import sys
from pathlib import Path
from unittest.mock import patch
import os

import joblib
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.b_frozen_head_frontier import read,require,save,sha
from scripts.run_b_shared_preference_pair import environment
from superagent.ab_methods import TEXT_DEEP,TEXT_EDIT
from superagent.ab_sentence import DeepPreferenceFeatures
from superagent.prediction_evidence import fingerprint


def run(path,workspace,out):
    arm=path/'deep-edit'
    # Wait for the main paired result to be sealed before running a diagnostic.
    result=sha(path/'results.json')
    history=read(arm/'history.json')
    records={(r['candidate_id'],r['split_id']):r for r in read(arm/'prediction_evidence.json')}
    measurements={}
    with patch.dict(os.environ,environment(workspace)):
        for r in history:
            if r['candidate']!=TEXT_EDIT or r['status']!='completed':
                continue
            split=r['split_id']
            f=arm/f"trial_{r['trial']}.joblib"
            model=joblib.load(f)['model']
            features=model.named_steps['features']
            require(isinstance(features,DeepPreferenceFeatures) and features.edit_context,'Unexpected trained representation')
            classifier=model.named_steps['classifier']
            payload=arm/(split+'.joblib')
            d=joblib.load(payload)
            y=np.array(d['y_search'])
            p=records[TEXT_EDIT,split]
            require(sha(payload)==p['evaluation_payload_sha256'] and fingerprint(y)==p['label_sha256'],'Misaligned diagnostic')
            tokens_before=features.usage_['input_tokens']
            x=features.transform(d['X_search'])
            margin=classifier.decision_function(x)
            np.testing.assert_array_equal((margin>0).astype(int),p['predictions'])
            reduced=np.asarray(x[:,:-384]@classifier.coef_[0,:-384]).ravel()+classifier.intercept_[0]
            full=(margin>0)==y
            masked=(reduced>0)==y
            active=np.asarray(abs(x[:,-384:]).sum(axis=1)).ravel()>0
            np.testing.assert_allclose(reduced[~active],margin[~active],atol=1e-10,rtol=0)
            record=dict(rows=len(y),active_rows=int(active.sum()),masked_correct=int(masked.sum()),full_correct=int(full.sum()),
                direct_edit_gains=int((full&~masked).sum()),direct_edit_losses=int((masked&~full).sum()),
                residual_coefficient_norm=float(np.linalg.norm(classifier.coef_[0,-384:])),
                local_frozen_encoder_tokens=features.usage_['input_tokens']-tokens_before,
                trained_weights=sha(f),evaluation_payload=sha(payload))
            if (TEXT_DEEP,split) in records:
                baseline=records[TEXT_DEEP,split]
                require(baseline['evaluation_payload_sha256']==p['evaluation_payload_sha256'] and
                        baseline['row_ids']==p['row_ids'],'Control rows differ')
                base=np.asarray(baseline['predictions'])==y
                record['plain_deep_correct']=int(base.sum())
                record['masked_vs_plain_net']=int(masked.sum()-base.sum())
            measurements[split]=record
    save(out,dict(paired_result_sha256=result,source_sha256=sha(Path(__file__)),measurements=measurements,
                  added_classifier_fits=0,model_api_calls=0,external_holdout_scored=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--workspace',type=Path,required=True)
    a=p.parse_args()
    run(a.run,a.workspace.resolve(),a.output)
