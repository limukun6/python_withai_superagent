"""Audit which measured errors and replications the native scheduler used."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
from scripts.b_frozen_head_frontier import paired,require,save,sha
from superagent.ab_edit_views import edit_view
from superagent.ab_methods import TEXT_DEEP,TEXT_EDIT
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups


def read(p):
    return json.loads(p.read_text())


def run(folder,out):
    summary=read(folder/'results.json')
    pairs={}
    paths={}
    for arm in ('deep','deep-edit'):
        run_dir=folder/arm
        history=read(run_dir/'history.json')
        records=read(run_dir/'prediction_evidence.json')
        observed={(r['candidate_id'],r['split_id']):r for r in records}
        choices=read(run_dir/'actions.json')
        result=read(run_dir/'confirmation.json')
        selected=read(run_dir/'run_outcome.json')['selected_candidate']
        paths[arm]=dict(selected=selected,fits=read(run_dir/'classifier_fits.json')['attempted_classifier_fits'],
            decisions=[dict(action=v['action'],candidate=v['candidate_id'],split=v['split_id'],
                            evidence=v['evidence_trials'],reason=v['reason']) for v in choices],
            confirmation_review=result.get('b_scheduler_review'),methods_seen=len({r['candidate'] for r in history}))
        if arm=='deep':
            continue
        for split in ('primary','confirmation_1','confirmation_2'):
            if not all((name,split) in observed for name in (TEXT_DEEP,TEXT_EDIT)):
                continue
            src=run_dir/(split+'.joblib')
            data=joblib.load(src)
            y=np.asarray(data['y_search'])
            rows=data['X_search']
            active=np.array([edit_view(*r)[1] for r in rows])
            g=sample_groups(SimpleNamespace(kind='pair_text',X_train=rows),'shared_text')
            hits=[]
            part=read(run_dir/(split+'_split.json'))
            for name in (TEXT_DEEP,TEXT_EDIT):
                p=observed[name,split]
                require(p['row_ids']==part['search_indices'] and p['evaluation_payload_sha256']==sha(src) and
                        p['label_sha256']==fingerprint(y) and p['prediction_sha256']==fingerprint(p['predictions']),
                        'Unverified comparable evidence')
                hits.append(np.array(p['predictions'])==y)
            a,b=hits
            gain=b.astype(int)-a
            slices={}
            for name,mask in (('active',active),('inactive',~active)):
                slices[name]=dict(rows=int(mask.sum()),deep_correct=int(a[mask].sum()),edit_correct=int(b[mask].sum()),
                    gained=int((gain[mask]>0).sum()),lost=int((gain[mask]<0).sum()))
            pairs[split]=dict(rows=len(y),deep_correct=int(a.sum()),edit_correct=int(b.sum()),slices=slices,
                              paired=paired(gain,g),payload_sha256=sha(src))
    save(out,dict(source=sha(Path(__file__)),results=sha(folder/'results.json'),paths=paths,
                  matched_internal_partitions=pairs,external_delta=summary['delta'],extra_fits=0,
                  model_api=0,selection_uses_outer=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('run',type=Path)
    p.add_argument('output',type=Path)
    a=p.parse_args()
    run(a.run,a.output)
