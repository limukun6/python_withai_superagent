"""Compare prior nominated research models with this pair's frozen exports on aligned rows."""
import argparse
import csv
from pathlib import Path
import numpy as np
from b_frozen_head_frontier import read,require,save,sha,paired


def run(pair,inputs,prior,out):
    seal=read(pair/'exports_frozen.json')
    scored=read(pair/'results.json')
    require(sha(pair/'exports_frozen.json')==scored['development_seal'],'Paired results not sealed')
    protocol=read(pair/'protocol.json')['interfaces']['files']
    require(sha(inputs/'groups_only.json')==protocol['groups_only.json'] and
            sha(inputs/'scoring_only.json')==protocol['scoring_only.json'],'Reserve identities differ')
    truth=read(inputs/'scoring_only.json')
    labels=np.asarray(truth['labels'])
    groups=np.asarray(read(inputs/'groups_only.json'))
    receipt=read(prior/'predictions.json')['blocks']['reserve']
    require(sha(prior/'reserve.npz')==receipt['sha256'],'Historical predictions changed')
    with np.load(prior/'reserve.npz') as old:
        np.testing.assert_array_equal(truth['original_rows'],old['row_ids'])
        historical={name:(old[name]==labels) for name in ('adapter','adapted_equal')}
    results={}
    for arm in ('deep','deep-edit'):
        f=pair/(arm+'_export_0')/'rows.csv'
        require(sha(f)==seal['predictions'][arm],'Export changed')
        with f.open() as stream:
            preds=list(csv.DictReader(stream))
        np.testing.assert_array_equal([int(v['row_id']) for v in preds],np.arange(len(labels)))
        correct=np.array([int(v['prediction']) for v in preds])==labels
        require(int(correct.sum())==scored['correct'][arm],'Outcome differs from primary study')
        results[arm]={}
        for name,old in historical.items():
            delta=correct.astype(int)-old
            results[arm][name]=dict(agent_correct=int(correct.sum()),historical_correct=int(old.sum()),
                agent_only_correct=int((correct&~old).sum()),historical_only_correct=int((~correct&old).sum()),
                same_wrong=int((~correct&~old).sum()),**paired(delta,groups))
    save(out,dict(rows=len(labels),contrasts=results,source=sha(Path(__file__)),old_score_seal=receipt['sha256'],
                  new_score_seal=sha(pair/'exports_frozen.json'),new_supervised_fits=0,model_api=0,
                  purpose='Post-selection comparison to two pre-existing developer-nominated models on the same development reserve'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('pair','inputs','prior','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    run(a.pair,a.inputs,a.prior,a.output)
