"""Use documented headers to identify a generic selector-to-measurement relationship."""
import argparse
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
import re
import time
import unicodedata

import numpy as np
import pandas as pd

from scripts.study_task1_candidates import ROLES, bound
from scripts.study_task1_measurements import interpret
from scripts.study_task1_tabm import elect
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.ab_sentence import ALTERNATIVES, encode_alternative
from superagent.attribute_alignment import comparable_fields
from superagent.field_interactions import choose_pairs, rank_pairs
from superagent.joint_context import ContextTokenTree, blocks, tokens
from superagent.models import _tabular_catboost

PLAN = ROOT / "research/task1_feedback_extension_1127/declared_semantic_binding.json"
METHODS = ("global", "pooled")


def normalize(s):
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", s)
    s = s.replace("<", " less than ").replace(">", " greater than ").replace("&", " and ")
    return " ".join(re.findall(r"[^\W\d_]+|\d+", s.casefold(), flags=re.UNICODE))


def lexical(a, b):
    x, y = a.split(), b.split()
    if not x or not y:
        return 0.
    overlap = len(set(x) & set(y)) / len(set(x) | set(y))
    letters = SequenceMatcher(None, "".join(x), "".join(y), autojunk=False).ratio()
    return .5*overlap + .5*letters


def discover(x, names, embed):
    families = comparable_fields(x, range(x.shape[1]))
    options = []
    for family in families:
        words = [normalize(names[j]) for j in family]
        if any(not value or re.fullmatch(r"(?:field|column|feature|col|var|x)?\s*\d+", value) for value in words):
            continue
        for gate, col in enumerate(x.T):
            if gate in family:
                continue
            values, counts = np.unique(col, return_counts=True)
            if not 2 <= len(values) <= 24:
                continue
            keys = [normalize(v) for v in values]
            if sum(any(c.isalpha() for c in k) for k in keys) < 2:
                continue
            vec = embed(words+keys)
            cosines = vec[len(words):] @ vec[:len(words)].T
            text = np.array([[lexical(k, word) for word in words] for k in keys])
            score = .52*text + .48*cosines
            table, accepted, margin_sum = {}, 0, 0.
            for i, v in enumerate(values):
                own = np.argsort(-score[i], kind="stable")
                best, runner = own[:2]
                margin = score[i, best] - score[i, runner]
                if score[i,best] >= .63 and margin >= .065 and (text[i,best] >= .25 or cosines[i,best] >= .78):
                    table[v] = family[best]
                    accepted += counts[i]
                    margin_sum += counts[i]*margin
            if len(set(table.values())) >= 2 and accepted >= .6*len(col):
                options.append((margin_sum/len(col), accepted/len(col), normalize(names[gate]), gate, family, table))
    if not options:
        return None
    # Names break ties to preserve equivalent decisions when the input columns are permuted.
    return max(options, key=lambda opt: (opt[0], opt[1], opt[2]))


class TextCache:
    def __init__(self):
        self.lookup = {}
        self.encoded_tokens = 0

    def __call__(self, words):
        missing = sorted(set(words)-self.lookup.keys())
        if missing:
            data, count = encode_alternative(missing, "bge")
            self.encoded_tokens += count
            self.lookup.update(zip(missing, data))
        return np.stack([self.lookup[w] for w in words])


def representations(x, base, binding):
    _, _, _, gate, family, mapping = binding
    cols = [mapping.get(v) for v in x[:,gate]]
    selected = np.asarray(["__MISSING__" if j is None else x[i,j] for i,j in enumerate(cols)])
    extras = np.column_stack((selected, tokens(np.column_stack((x[:,gate], selected)))))
    cats = np.column_stack((base.transform(x), extras))
    def quantity(v):
        parsed = interpret(v)
        return parsed[1] if parsed else np.nan
    picked = np.asarray([quantity(v) for v in selected])
    measurements = np.array([[quantity(v) for v in row] for row in x[:,family]])
    counts = np.isfinite(measurements).sum(1)
    average = np.divide(np.nansum(measurements,axis=1), np.maximum(counts,1))
    numeric = np.column_stack((picked,picked-average,np.where(counts,average,np.nan)))
    return pd.concat((pd.DataFrame(cats),pd.DataFrame(numeric)),axis=1,ignore_index=True), cats.shape[1]


def run(out, task_spec):
    start = time.monotonic()
    spec = read(task_spec)
    evidence = spec["declaration_evidence"]["fields"]["columns"]
    names = [c["name"] for c in spec["input_schema"]["columns"]]
    require(names == evidence["value"] and evidence["status"] == "confirmed", "Unconfirmed feature names")
    info = read(out/"inputs.json")
    fitting = bound(out,"fitting.joblib",info)
    x,y = fitting["x"], fitting["y"]
    panels = [bound(out,r+"_features.joblib",info) for r in ROLES]
    query = np.concatenate(panels)
    require(len(names) == x.shape[1] and len(set(names)) == len(names), "Misaligned declared schema")
    encoder = TextCache()
    binding = discover(x,names,encoder)
    require(binding is not None, "No supported semantic binding")
    _, coverage, _, gate, family, table = binding
    permutation = np.random.default_rng(18607).permutation(len(names))
    permuted = discover(x[:,permutation],[names[i] for i in permutation],encoder)
    require(permuted is not None and permutation[permuted[3]] == gate
            and {v:permutation[j] for v,j in permuted[5].items()} == table, "Binding changes under column reordering")
    require(discover(x,["field "+str(i) for i in range(len(names))],encoder) is None, "Arbitrary numbered headers matched")
    base = ContextTokenTree(dict(pair_limit=3))
    base.profile, base.context = blocks(x)
    base.pairs = choose_pairs(rank_pairs(x,y),3)
    train,categorical_count = representations(x,base,binding)
    test,_ = representations(query,base,binding)
    cats = list(range(categorical_count))
    global_tree = _tabular_catboost(y,6,3.)
    global_tree.set_params(thread_count=5)
    global_tree.fit(train,y,cat_features=cats)
    global_prob = global_tree.predict_proba(test)
    pooled = global_prob.copy()
    specialists = []
    for val, n in Counter(x[:,gate]).most_common(8):
        fit = x[:,gate] == val
        predict = query[:,gate] == val
        if val not in table or n < 192 or len(np.unique(y[fit])) != len(global_tree.classes_):
            continue
        tree = _tabular_catboost(y[fit],4,3.)
        tree.set_params(thread_count=5,iterations=400,learning_rate=.045)
        tree.fit(train.loc[fit],y[fit],cat_features=cats)
        require(np.array_equal(tree.classes_,global_tree.classes_), "Changed class order")
        if predict.any():
            weight = n/(n+500.)
            pooled[predict] = (1-weight)*global_prob[predict] + weight*tree.predict_proba(test.loc[predict])
        specialists.append(n)
    for name, output in zip(METHODS,(global_prob,pooled)):
        folder = out/("binding_"+name)
        folder.mkdir(exist_ok=False)
        require(np.isfinite(output).all() and np.allclose(output.sum(1),1.),"Invalid binding probability")
        hashes = {}
        for role, probabilities in zip(ROLES,np.split(output,np.cumsum([len(p) for p in panels])[:-1])):
            path = folder/(role+".npy")
            np.save(path,probabilities,allow_pickle=False)
            hashes[role] = sha(path)
        record = dict(protocol=sha(PLAN),inputs=sha(out/"inputs.json"),declared_spec_sha256=sha(task_spec),
            classes=global_tree.classes_.tolist(),predictions=hashes,additional_classifier_fits=(1 if name=="global" else len(specialists)),
            family_size=len(family),selector_coverage=float(coverage),distinct_bound_fields=len(set(table.values())),
            specialists=list(map(int,specialists)) if name=="pooled" else [],shared_frozen_encoder=ALTERNATIVES["bge"]["model"],
            local_text_tokens_shared=encoder.encoded_tokens,anonymous_abstention=True,permutation_invariant=True,
            seconds_both=round(time.monotonic()-start,2),api_requests=0)
        save(folder/"sealed.json",record)
        print(name,record["family_size"],record["selector_coverage"],record["additional_classifier_fits"],flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("run","elect"))
    parser.add_argument("--out",required=True,type=lambda p:Path(p).resolve())
    parser.add_argument("--task-spec",type=lambda p:Path(p).resolve())
    args=parser.parse_args()
    if args.action=="run":
        run(args.out,args.task_spec)
    else:
        elect(args.out,METHODS,"binding",PLAN)
