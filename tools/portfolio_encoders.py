"""Frozen downstream encoders with the unified preference head and optional feature-only cache."""
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from superagent.ab_sentence import SentenceFeatures, _encoder, encode_frozen
from tools.encoder_assets import BGE, MPNET, ROBERTA, MODELS, verified


def weights_dir():
    folder = os.environ.get("SUPERAGENT_PORTFOLIO_WEIGHTS")
    if not folder:
        raise ValueError("Set SUPERAGENT_PORTFOLIO_WEIGHTS to the directory containing the pinned model folders")
    return Path(folder)


@lru_cache(maxsize=2)
def transformer(folder, name):
    import torch
    from tokenizers import Tokenizer
    from transformers import MPNetModel, RobertaModel

    tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=384 if name == MPNET else 512)
    cls = MPNetModel if name == MPNET else RobertaModel
    model, info = cls.from_pretrained(folder, add_pooling_layer=False, use_safetensors=True,
                                      local_files_only=True, output_loading_info=True)
    if info["missing_keys"] or info["mismatched_keys"] or info["error_msgs"]:
        raise ValueError(f"Incomplete {name} encoder: {info['missing_keys']}")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return tokenizer, model.eval().requires_grad_(False)


def encode_transformer(folder, name, items):
    import torch
    tokenizer, model = transformer(folder, name)
    pair = name == ROBERTA
    batch_inputs = [*items, *((b, a) for a, b in items)] if pair else items
    enc = tokenizer.encode_batch(batch_inputs)
    lengths = np.asarray([len(item.ids) for item in enc])
    order = np.argsort(lengths, kind="stable")
    vectors = np.empty((len(enc), model.config.hidden_size), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(order), 32):
            ix = order[start:start+32]
            ids = torch.full((len(ix), int(lengths[ix].max())), model.config.pad_token_id, dtype=torch.long)
            mask = torch.zeros_like(ids)
            sides = torch.full_like(ids, -1) if pair else None
            for i, j in enumerate(ix):
                seq = enc[j]
                count = len(seq.ids)
                ids[i, :count] = torch.tensor(seq.ids)
                mask[i, :count] = 1
                if pair:
                    sides[i, :count] = torch.tensor([-1 if side is None else side for side in seq.sequence_ids])
            hidden = model(input_ids=ids, attention_mask=mask).last_hidden_state
            if pair:
                left, right = (sides == 0).unsqueeze(2), (sides == 1).unsqueeze(2)
                mean = (hidden*left).sum(1) / left.sum(1).clamp(min=1) - (hidden*right).sum(1) / right.sum(1).clamp(min=1)
            else:
                mean = (hidden*mask.unsqueeze(2)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
            vectors[ix] = torch.nn.functional.normalize(mean, dim=1).numpy()
    if pair:
        n = len(items)
        vectors = (vectors[:n] - vectors[n:]) / 2
    if not np.isfinite(vectors).all():
        raise ValueError(f"Non-finite {name} features")
    return vectors


def compute(name, keys):
    folder = verified(weights_dir(), name)
    if name == BGE:
        tok, model = _encoder(folder, 512)
        return encode_frozen(keys, tok, model, pool="cls")[0]
    values = [tuple(json.loads(k)) for k in keys] if name == ROBERTA else keys
    return encode_transformer(folder, name, values)


def pair_key(a, b):
    return json.dumps([a, b] if a <= b else [b, a], ensure_ascii=False, separators=(",", ":"))


def keys_for(name, rows):
    if name == ROBERTA:
        return [pair_key(a, b) for a, b in rows]
    return [text for pair in rows for text in pair]


def bank_signature(name):
    return {"model": name, "revision": MODELS[name]["revision"],
            "source": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def build_bank(name, rows, folder):
    folder.mkdir(parents=True, exist_ok=True)
    output = folder / (name + ".npz")
    keys = sorted(set(keys_for(name, rows)))
    if output.exists():
        lookup, _ = load_bank(output, name, output.stat().st_mtime_ns)
        if set(lookup) != set(keys):
            raise ValueError("Existing feature bank has different training inputs")
        return output
    features = compute(name, keys)
    metadata = json.dumps({"signature": bank_signature(name), "keys": keys}, ensure_ascii=False).encode()
    with NamedTemporaryFile(dir=folder, prefix=".features-", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with tmp_path.open("wb") as buf:
            np.savez_compressed(buf, metadata=np.frombuffer(metadata, dtype=np.uint8), vectors=features)
        os.link(tmp_path, output)
    finally:
        tmp_path.unlink(missing_ok=True)
    return output


@lru_cache(maxsize=4)
def load_bank(path, name, modified):
    with np.load(path, allow_pickle=False) as array:
        meta = json.loads(array["metadata"].tobytes().decode())
        vectors = array["vectors"]
    keys = meta["keys"]
    if (meta["signature"] != bank_signature(name) or len(keys) != len(set(keys))
            or vectors.shape != (len(keys), 384 if name == BGE else 768)
            or not np.isfinite(vectors).all()):
        raise ValueError("Frozen feature cache identity, shape or values differ")
    return {k: i for i, k in enumerate(keys)}, vectors


def frozen_vectors(name, keys):
    cache = os.environ.get("SUPERAGENT_PORTFOLIO_CACHE")
    cached_path = Path(cache) / (name + ".npz") if cache else None
    lookup, old = ({}, None)
    if cached_path and cached_path.is_file():
        lookup, old = load_bank(cached_path, name, cached_path.stat().st_mtime_ns)
    unseen = list(dict.fromkeys(k for k in keys if k not in lookup))
    if not unseen:
        return old[[lookup[k] for k in keys]]
    new = compute(name, unseen)
    new_lookup = {k: i for i, k in enumerate(unseen)}
    return np.stack([old[lookup[k]] if k in lookup else new[new_lookup[k]] for k in keys])


class PortfolioFeatures(SentenceFeatures):
    def __init__(self, name):
        super().__init__("pair_text")
        self.name = name

    def semantic_blocks(self, rows, texts):
        values = frozen_vectors(self.name, keys_for(self.name, rows))
        if self.name == ROBERTA:
            sign = np.asarray([int(a < b) - int(a > b) for a, b in rows], dtype=np.float32)
            return [csr_matrix(values * sign[:, None])], 0
        return self.embedding_features(values), 0


def fit_head(name, rows, labels):
    y = np.asarray(labels)
    if not np.array_equal(np.unique(y), [0, 1]) or len(rows) != len(y):
        raise ValueError("An aligned binary preference training set is required")
    features = PortfolioFeatures(name)
    x = features.fit_transform(rows)
    classifier = LogisticRegression(C=.3, max_iter=2000, random_state=42)
    classifier.fit(vstack([x, -x]), np.r_[y, 1-y])
    model = Pipeline([("features", features), ("classifier", classifier)])
    model.training_metadata = {"frozen_downstream_encoder": MODELS[name]["repo"],
                               "revision": MODELS[name]["revision"], "classifier_fits": 1}
    return model
