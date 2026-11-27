"""Precompute immutable per-text features from a training partition for local studies."""
import argparse
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import numpy as np

from superagent.ab_sentence import ALTERNATIVES
from superagent.paired_encoder import VARIANTS, bank_name, encode_texts, text_key


def configure_weights(weights):
    os.environ["SUPERAGENT_DEEP_MINILM_WEIGHTS"] = str(weights / "all-MiniLM-L12-v2")
    for variant in VARIANTS:
        spec = ALTERNATIVES[variant]
        os.environ[spec["env"]] = str(weights / spec["model"].split("/")[-1])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("training-payload", "cache", "weights"):
        p.add_argument("--"+name, type=lambda s:Path(s).resolve(), required=True)
    p.add_argument("--encoder", choices=(*VARIANTS,"minilm_l12"), required=True)
    args = p.parse_args()
    configure_weights(args.weights)
    payload = joblib.load(args.training_payload)
    texts = sorted({s for part in ("X_train", "X_search") for row in payload[part] for s in row})
    args.cache.mkdir(parents=True, exist_ok=True)
    path = args.cache / bank_name(args.encoder)
    keys = np.array([text_key(text) for text in texts])
    if path.exists():
        with np.load(path, allow_pickle=False) as previous:
            if not np.array_equal(previous["keys"], keys):
                raise ValueError("Existing bank belongs to another text set")
    else:
        vectors = encode_texts(args.encoder, texts)
        if not np.isfinite(vectors).all():
            raise ValueError("Nonfinite encoder output")
        with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as temp:
            temp_name = Path(temp.name)
            np.savez(temp, keys=keys, vectors=vectors)
        try:
            os.link(temp_name, path)
        finally:
            temp_name.unlink(missing_ok=True)
    print(f"{args.encoder}: {len(keys)} distinct training texts, {path.stat().st_size} bytes",flush=True)
