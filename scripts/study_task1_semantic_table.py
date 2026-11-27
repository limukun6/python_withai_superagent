"""Reuse the existing offline sentence encoders as independent table representations."""
import argparse
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.study_task1_candidates import ROLES, bound
from scripts.study_task1_tabm import elect
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.ab_sentence import ALTERNATIVES, encode_alternative
from superagent.field_interactions import choose_pairs, rank_pairs
from superagent.joint_context import ContextTokenTree, blocks, tokens
from superagent.models import _tabular_catboost

PLAN = ROOT / "research/task1_feedback_extension_1127/semantic_table_views.json"
VARIANTS = ("e5", "bge")
HEADS = ("tree", "linear")
METHODS = tuple(v+"_"+h for v in VARIANTS for h in HEADS)


def descriptions(x, fields):
    return [". ".join(f"Field {j+1} is {value}" for j, value in zip(fields, row)) for row in x[:, fields]]


def encode_blocks(x, query, fields, variant):
    fit_parts, query_parts, fit_keys, query_keys = [], [], [], []
    token_total = 0
    for columns in fields:
        train_text = descriptions(x, columns)
        query_text = descriptions(query, columns)
        vectors, local_tokens = encode_alternative(train_text + query_text, variant)
        # Frozen models encode each text independently. Learn all dataset-dependent transforms on training rows.
        token_total += local_tokens
        n = len(x)
        component_count = min(16, n-1, vectors.shape[1], len(set(train_text))-1)
        require(component_count > 0, "A semantic view was constant")
        transform = PCA(n_components=component_count, svd_solver="randomized", random_state=9321)
        fit_parts.append(transform.fit_transform(vectors[:n]))
        query_parts.append(transform.transform(vectors[n:]))
        fit_keys.append(tokens(x[:, columns]))
        query_keys.append(tokens(query[:, columns]))
    return fit_parts, query_parts, np.column_stack(fit_keys), np.column_stack(query_keys), token_total


def linear_inputs(fit, query, fit_keys, query_keys):
    train, test = [], []
    for a,b in zip(fit, query):
        scaler = StandardScaler()
        train.append(scaler.fit_transform(a))
        test.append(scaler.transform(b))
    def expand(parts):
        left,right = parts
        interactions = (left[:, :, None] * right[:, None, :]).reshape(len(left), -1)
        return sparse.csr_matrix(np.column_stack((left, right, interactions / np.sqrt(interactions.shape[1]))))
    encoder = OneHotEncoder(handle_unknown="ignore")
    return (sparse.hstack((expand(train), encoder.fit_transform(fit_keys)), format="csr"),
            sparse.hstack((expand(test), encoder.transform(query_keys)), format="csr"))


def run(out, variant):
    info = read(out / "inputs.json")
    fitting = bound(out, "fitting.joblib", info)
    x,y = fitting["x"], fitting["y"]
    panels = [bound(out, r+"_features.joblib", info) for r in ROLES]
    query = np.concatenate(panels)
    fields = blocks(x)
    start = time.monotonic()
    left,right,fit_keys,query_keys,tokens_used = encode_blocks(x,query,fields,variant)
    encode_seconds = round(time.monotonic()-start, 2)
    for head in HEADS:
        folder = out / ("semantic_" + variant + "_" + head)
        folder.mkdir(exist_ok=False)
        start = time.monotonic()
        if head == "tree":
            base = ContextTokenTree(dict(pair_limit=3))
            base.profile,base.context = fields
            base.pairs = choose_pairs(rank_pairs(x,y),3)
            def combine(data, vectors):
                cats = base.transform(data)
                return pd.concat([pd.DataFrame(cats),pd.DataFrame(np.column_stack(vectors))], axis=1, ignore_index=True), cats.shape[1]
            train,num_categories = combine(x,left)
            test,_ = combine(query,right)
            model = _tabular_catboost(y,6,3.)
            model.set_params(thread_count=5)
            model.fit(train,y,cat_features=list(range(num_categories)))
        else:
            train,test = linear_inputs(left,right,fit_keys,query_keys)
            model = LogisticRegression(C=1., solver="liblinear", max_iter=500, random_state=9321).fit(train,y)
        output = np.asarray(model.predict_proba(test),dtype=float)
        require(np.isfinite(output).all() and np.allclose(output.sum(1),1.),"Invalid semantic probabilities")
        hashes = {}
        for role, probs in zip(ROLES,np.split(output,np.cumsum([len(p) for p in panels])[:-1])):
            path = folder / (role + ".npy")
            np.save(path,probs,allow_pickle=False)
            hashes[role] = sha(path)
        candidate = ALTERNATIVES[variant]
        details = dict(protocol=sha(PLAN),inputs=sha(out / "inputs.json"),classes=model.classes_.tolist(),
                       supervised_fits=1,predictions=hashes,model=candidate["model"],revision=candidate["revision"],
                       weights_sha256=candidate["files"]["model.safetensors"][1],frozen=True,
                       local_encoder_tokens_shared=tokens_used,encoding_seconds_shared=encode_seconds,
                       class_head_seconds=round(time.monotonic()-start,2),api_requests=0)
        save(folder/"sealed.json",details)
        print(variant,head,encode_seconds,details["class_head_seconds"],flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=(*VARIANTS,"elect"))
    parser.add_argument("--out", required=True, type=lambda p: Path(p).resolve())
    args = parser.parse_args()
    if args.action == "elect":
        elect(args.out,METHODS,"semantic",PLAN)
    else:
        run(args.out,args.action)
