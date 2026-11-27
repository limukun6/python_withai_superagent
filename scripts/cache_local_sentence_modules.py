"""Warm only frozen features for the training package; no supervised training or remote inference."""
import argparse
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np

from superagent.external_assets import activate_resource_directory
from superagent.local_sentence import bank_path,encode,installed
from superagent.paired_encoder import text_key


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('weights','training-payload','cache'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    activate_resource_directory(args.weights)
    partitions=joblib.load(args.training_payload)
    texts=sorted({s for k in ('X_train','X_search') for row in partitions[k] for s in row})
    keys=np.asarray([text_key(s) for s in texts])
    args.cache.mkdir(parents=True,exist_ok=True)
    for s in installed():
        path=bank_path(args.cache,s)
        if path.exists():
            with np.load(path,allow_pickle=False) as z:
                if not np.array_equal(z['keys'],keys) or z['vectors'].shape!=(len(keys),s['width']):
                    raise ValueError('Existing unsupervised cache belongs to a different study')
        else:
            out=encode(s,texts)
            with NamedTemporaryFile(dir=path.parent,delete=False) as temp:
                name=Path(temp.name)
                np.savez(temp,keys=keys,vectors=out)
            try:
                os.link(name,path)
            finally:
                name.unlink(missing_ok=True)
        print(s['family'],s['signature'][:20],len(keys),'training texts',flush=True)
