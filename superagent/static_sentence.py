"""Read standard Sentence Transformers static embedding bags from portable data files."""
from functools import lru_cache

import numpy as np


def layout(root, parts):
    from safetensors import SafetensorError, safe_open

    if parts[0]["path"] in ("", "."):
        prefix = ""
    else:
        path = parts[0]["path"]
        if not isinstance(path, str) or not path.replace("_", "").isalnum():
            raise ValueError("Invalid static embedding directory")
        prefix = path + "/"
    if len(parts) == 2 and parts[1]["path"] not in ("", ".", "1_Normalize"):
        raise ValueError("Unsupported normalization directory")
    filename = prefix + "model.safetensors"
    try:
        with safe_open(root/filename, framework="numpy") as tensors:
            keys = tensors.keys()
            if keys not in (["embeddings"], ["embedding.weight"]):
                raise ValueError("Expected one static embedding table")
            table = tensors.get_slice(keys[0])
            shape = table.get_shape()
            if len(shape)!=2 or min(shape)<1 or table.get_dtype() not in ("F16","F32","F64"):
                raise ValueError("Invalid static embedding dimensions or dtype")
    except SafetensorError as exc:
        raise ValueError("Unreadable static embedding data") from exc
    return prefix, shape[1]


@lru_cache(maxsize=2)
def load_table(root, prefix, signature):
    from safetensors.numpy import load_file
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(root/(prefix+"tokenizer.json")))
    tokenizer.no_padding()
    tensors = load_file(root/(prefix+"model.safetensors"))
    table = next(iter(tensors.values()))
    if tokenizer.get_vocab_size() != len(table) or not np.isfinite(table).all():
        raise ValueError("Static tokenizer and embedding table are incompatible")
    return tokenizer, table


def encode(root, spec, texts):
    tokenizer, table = load_table(root, spec["module_prefix"], spec["signature"])
    output = np.zeros((len(texts),spec["width"]), dtype=np.float32)
    for start in range(0,len(texts),256):
        batch = tokenizer.encode_batch(texts[start:start+256], add_special_tokens=False)
        for row, tokens in enumerate(batch,start):
            if tokens.ids:
                output[row] = table[tokens.ids].mean(axis=0,dtype=np.float32)
    if spec["normalize"]:
        output /= np.maximum(np.linalg.norm(output,axis=1,keepdims=True),1e-12)
    return output
