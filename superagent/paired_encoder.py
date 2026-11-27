"""Frozen sentence representations and symmetric training scales for preference tasks."""
from functools import lru_cache, partial
import hashlib
import os
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.linear_model import LogisticRegression

from .ab_preference_output import CanonicalPreferencePipeline
from .ab_sentence import ALTERNATIVES, AlternativeSentenceFeatures, _encoder, encode_frozen, verified_files
from .models import Candidate
from .word_familiarity import WordFamiliarity


VARIANTS = ("gte", "electra")
NAMES = {f"{variant}_paired_{head}": (variant, head)
         for variant in VARIANTS for head in ("linear", "quadratic")}


@lru_cache(maxsize=1)
def discriminator(root):
    import torch
    from tokenizers import Tokenizer
    from transformers import ElectraForPreTraining
    tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.enable_truncation(max_length=512)
    model, info = ElectraForPreTraining.from_pretrained(root, local_files_only=True, use_safetensors=False,
                                                        weights_only=True, output_loading_info=True)
    if info["missing_keys"] or info["mismatched_keys"] or info["error_msgs"]:
        raise ValueError("Incomplete ELECTRA weights")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return tokenizer, model.eval().requires_grad_(False)


def encode_texts(variant, texts):
    import torch
    spec = ALTERNATIVES[variant]
    root = verified_files(spec["env"], spec["files"])
    if root is None:
        raise ValueError("Missing pinned weights for " + variant)
    if variant == "gte":
        tokenizer, model = _encoder(root, 512)
        return encode_frozen(texts, tokenizer, model)[0]
    if variant != "electra":
        raise ValueError("Unknown encoder")
    tokenizer, model = discriminator(root)
    encodings = tokenizer.encode_batch(texts)
    order = sorted(range(len(encodings)), key=lambda i: len(encodings[i].ids))
    vectors = np.zeros((len(texts), model.config.hidden_size + 4), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(order), 48):
            indices = order[start:start+48]
            width = len(encodings[indices[-1]].ids)
            ids = torch.full((len(indices), width), model.config.pad_token_id, dtype=torch.long)
            attention = torch.zeros_like(ids)
            for i, j in enumerate(indices):
                tokens = encodings[j].ids
                ids[i, :len(tokens)] = torch.as_tensor(tokens)
                attention[i, :len(tokens)] = 1
            hidden = model.electra(input_ids=ids, attention_mask=attention).last_hidden_state
            pooled = (hidden * attention.unsqueeze(-1)).sum(1) / attention.sum(1, keepdim=True)
            vectors[indices, :-4] = torch.nn.functional.normalize(pooled, dim=1).numpy()
            logits = model.discriminator_predictions(hidden).reshape(len(indices), width).numpy()
            for i, j in enumerate(indices):
                n = len(encodings[j].ids)
                ordinary = np.asarray(encodings[j].special_tokens_mask) == 0
                values = logits[i, :n][ordinary]
                if len(values):
                    vectors[j, -4:] = (values.mean(), values.std(), values.max(), np.quantile(values, .9))
    return vectors


def text_key(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bank_name(variant):
    return variant + "-" + ALTERNATIVES[variant]["revision"] + "-pool-v1.npz"


@lru_cache(maxsize=4)
def read_bank(path, modified):
    with np.load(path, allow_pickle=False) as archive:
        keys, vectors = archive["keys"], archive["vectors"]
    dimension = 384 if path.name.startswith("gte-") else 260
    if vectors.shape != (len(keys), dimension) or not np.isfinite(vectors).all():
        raise ValueError("Invalid frozen feature bank")
    return dict(zip(keys.tolist(), vectors))


def vectors_for(variant, texts):
    directory = os.environ.get("SUPERAGENT_PAIRED_FEATURE_CACHE")
    unique = list(dict.fromkeys(texts))
    indexed = {}
    if directory:
        path = Path(directory) / bank_name(variant)
        if path.is_file():
            bank = read_bank(path, path.stat().st_mtime_ns)
            indexed = {s: bank[k] for s in unique if (k := text_key(s)) in bank}
    missing = [s for s in unique if s not in indexed]
    if missing:
        indexed.update(zip(missing, encode_texts(variant, missing)))
    return np.stack([indexed[s] for s in texts])


class PairedEncoderFeatures(AlternativeSentenceFeatures):
    def __init__(self, variant, head):
        super().__init__("pair_text", variant)
        self.head = head

    def fit(self, rows, y=None):
        super().fit(rows, y)
        texts = list(dict.fromkeys(s for pair in rows for s in pair))
        vectors = vectors_for(self.variant, texts)
        self.center_ = vectors.mean(axis=0)
        self.scale_ = np.maximum(vectors.std(axis=0), 1e-4)
        self.familiarity_ = WordFamiliarity().fit(texts)
        distances = self.familiarity_.differences(rows)
        self.familiarity_scale_ = float(np.sqrt(np.mean(distances ** 2))) or 1.
        return self

    def semantic_blocks(self, rows, texts):
        dense = vectors_for(self.variant, texts)
        z = np.clip((dense - self.center_) / self.scale_, -6., 6.)
        left, right = z[::2], z[1::2]
        delta = left - right
        scale = np.sqrt(z.shape[1])
        blocks = [csr_matrix(delta / scale)]
        if self.head == "quadratic":
            blocks.append(csr_matrix(delta * (left + right) / (2. * scale)))
        familiarity = self.familiarity_.differences(rows) / self.familiarity_scale_
        blocks.append(csr_matrix(familiarity[:, None]))
        return blocks, 0


def fit_ranker(rows, labels, query, *, variant, head, regularization=.3):
    labels = np.asarray(labels)
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError("Pair preferences require a binary side label")
    features = PairedEncoderFeatures(variant, head).fit(rows)
    matrix = features.transform(rows)
    classifier = LogisticRegression(C=regularization, fit_intercept=False, max_iter=2000, random_state=42)
    classifier.fit(vstack([matrix, -matrix]), np.r_[labels, 1-labels])
    model = CanonicalPreferencePipeline([("features", features), ("classifier", classifier)])
    return model, model.predict(query)


def candidates():
    result = []
    for name, (variant, head) in NAMES.items():
        spec = ALTERNATIVES[variant]
        if verified_files(spec["env"], spec["files"]) is None:
            continue
        params = dict(encoder=spec["model"], revision=spec["revision"], pooling=spec["pool"],
                      representation=f"word_char_{variant}_scaled_{head}_familiarity", C=.3,
                      classifier="logistic", swap_augmentation=True)
        result.append(Candidate(name, params, partial(fit_ranker, variant=variant, head=head)))
    return result
