"""Compute a public masked encoder's independent numerical features for unlabelled training texts."""
import argparse
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np

from superagent.masked_context import CACHE_NAME,SPEC,encode,text_key
from superagent.ab_sentence import pinned_folder

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","weights"):
        p.add_argument("--"+name,required=True,type=lambda v:Path(v).resolve())
    args=p.parse_args()
    source=args.root/"raw/race/run/agent/primary.joblib"
    payload=joblib.load(source)
    texts=sorted({s for part in ("X_train","X_search") for row in payload[part] for s in row})
    directory=args.root/"feature_bank"
    target=directory/CACHE_NAME
    keys=np.asarray([text_key(s) for s in texts])
    folder=pinned_folder(args.weights/SPEC["model"].split("/")[-1],SPEC["files"])
    if folder is None:
        raise ValueError("Missing verified local masked encoder")
    if target.is_file():
        with np.load(target,allow_pickle=False) as saved:
            if not np.array_equal(saved["keys"],keys):
                raise ValueError("Existing vectors do not match these training texts")
    else:
        with NamedTemporaryFile(dir=directory,delete=False,suffix=".npz") as temp:
            temp_name=Path(temp.name)
        try:
            vectors=encode(texts,folder)
            np.savez(temp_name,keys=keys,vectors=vectors)
            os.link(temp_name,target)
        finally:
            temp_name.unlink(missing_ok=True)
    print("frozen vectors",len(texts),target.stat().st_size,flush=True)
