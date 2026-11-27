"""Store pinned contextual token features for the training-only preference study."""
import argparse
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np
import torch

from scripts.cache_paired_encoders import configure_weights
from superagent.ab_sentence import ALTERNATIVES, _encoder, verified_files
from superagent.paired_encoder import text_key


def build(payload, weights, destination):
    if destination.exists():
        raise FileExistsError(destination)
    configure_weights(weights)
    spec = ALTERNATIVES["gte"]
    root = verified_files(spec["env"], spec["files"])
    if root is None:
        raise ValueError("Pinned encoder unavailable")
    tokenizer, encoder = _encoder(root,512)
    data = joblib.load(payload)
    texts = sorted({text for name in ("X_train","X_search") for pair in data[name] for text in pair})
    encoded = tokenizer.encode_batch(texts)
    keys = np.array([text_key(s) for s in texts])
    selected = [np.flatnonzero(np.asarray(item.special_tokens_mask)==0) for item in encoded]
    selected = [indices if len(indices) else np.array([0]) for indices in selected]
    offsets = np.r_[0,np.cumsum([len(indices) for indices in selected])]
    vectors = np.empty((offsets[-1],encoder.config.hidden_size),dtype=np.float16)
    order = sorted(range(len(texts)),key=lambda i:len(encoded[i].ids))
    with torch.inference_mode():
        for start in range(0,len(order),48):
            batch = order[start:start+48]
            tokens = torch.zeros((len(batch),len(encoded[batch[-1]].ids)),dtype=torch.long)
            mask = torch.zeros_like(tokens)
            for row,index in enumerate(batch):
                length = len(encoded[index].ids)
                tokens[row,:length] = torch.tensor(encoded[index].ids)
                mask[row,:length] = 1
            hidden = encoder(input_ids=tokens,attention_mask=mask).last_hidden_state.numpy()
            for row,index in enumerate(batch):
                vectors[offsets[index]:offsets[index+1]] = hidden[row,selected[index]]
    if not np.isfinite(vectors).all():
        raise ValueError("Nonfinite token states")
    destination.parent.mkdir(parents=True,exist_ok=True)
    with NamedTemporaryFile(dir=destination.parent,suffix=".npz",delete=False) as output:
        temp=Path(output.name)
        np.savez(output, keys=keys, offsets=offsets, vectors=vectors, revision=spec["revision"])
    try:
        os.link(temp,destination)
    finally:
        temp.unlink(missing_ok=True)
    print(len(texts),len(vectors),destination.stat().st_size,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for n in ("training-payload","weights","output"):
        p.add_argument("--"+n,type=lambda s:Path(s).resolve(),required=True)
    args=p.parse_args()
    build(args.training_payload,args.weights,args.output)
