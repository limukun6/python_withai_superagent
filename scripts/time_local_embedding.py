"""Estimate native local embedding time from an ordered sample of training texts."""
import argparse
import hashlib
from pathlib import Path
import time

import joblib
import numpy as np

from scripts.unified_released_study import save
from superagent.external_assets import activate_resource_directory
from superagent.local_sentence import encode,installed


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('payload','weights','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    activate_resource_directory(args.weights)
    candidates=[s for s in installed() if s['pooling']=='last']
    if len(candidates)!=1:
        raise ValueError('Expected one newly installed last-token resource')
    spec=candidates[0]
    partition=joblib.load(args.payload)
    pool={s for field in ('X_train','X_search') for row in partition[field] for s in row}
    data=sorted(pool,key=lambda s:hashlib.sha256(('local-embedding-time-v1/'+s).encode()).digest())[:96]
    start=time.perf_counter()
    values,tokens=encode(spec,data,return_tokens=True)
    elapsed=time.perf_counter()-start
    # Include startup in the projection, which favors a conservative cost estimate.
    forecast=elapsed*len(pool)/len(data)
    result=dict(signature=spec['signature'],rows=len(data),distinct_training_texts=len(pool),tokens=tokens,
                seconds=elapsed,estimated_full_seconds=forecast,dimensions=values.shape[1],
                mean_norm=float(np.linalg.norm(values,axis=1).mean()),use_full_training=forecast<=3600)
    save(args.output,result)
    print(result,flush=True)
