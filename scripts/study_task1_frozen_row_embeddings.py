"""Train downstream heads on a frozen row encoder with a disjoint labeled reference bank."""
import argparse
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from scripts.study_task1_adaptive_memory import cached_probability
from scripts.study_task1_candidates import ROLES, bound
from scripts.study_task1_cross_context import fit_offset
from scripts.study_task1_tabm import elect
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.models import _clean_categories, _tabular_catboost
from superagent.splits import sample_groups
from superagent.table_foundation import SHA256

PLAN = ROOT / "research/task1_feedback_extension_1127/frozen_row_embeddings.json"
METHODS = ("augmented_linear", "augmented_tree", "offset")


def extract(model, rows):
    hidden = []
    def capture(module, args, output):
        hidden.append(output.detach().float().cpu().numpy().copy())
    hook = model.model_.row_interactor.register_forward_hook(capture)
    try:
        p = model.predict_proba(pd.DataFrame(rows, dtype=object))
    finally:
        hook.remove()
    require(len(hidden) == 1 and hidden[0].ndim == 3 and hidden[0].shape[:2] == (1,len(rows)),
            "Expected exactly one frozen query-row view")
    return hidden[0][0], p


def prepare(out, weight):
    from tabicl import TabICLClassifier
    import torch
    require(version("tabicl") == "2.0.3" and sha(weight) == SHA256, "Unexpected frozen encoder")
    info = read(out / "inputs.json")
    data = bound(out, "fitting.joblib", info)
    x, y = _clean_categories(data["x"]), np.asarray(data["y"])
    groups = sample_groups(SimpleNamespace(kind="categorical_tabular", X_train=x))
    head, ref = next(StratifiedGroupKFold(5, shuffle=True, random_state=38317).split(x,y,groups))
    require(set(groups[head]).isdisjoint(groups[ref]) and len(np.unique(y[ref])) == 2, "Reference leaks into head training")
    queries = np.concatenate([x[head], *[_clean_categories(bound(out, r + "_features.joblib",info)) for r in ROLES]])
    area = out / "frozen_row_representations"
    area.mkdir(exist_ok=False)
    model = TabICLClassifier(n_estimators=1, batch_size=1, model_path=weight, allow_auto_download=False,
              device="mps", use_amp=False, use_fa3=False, kv_cache=True, n_jobs=1, random_state=173)
    torch.set_num_threads(2)
    save(area / "launch.json", dict(protocol=sha(PLAN), inputs=sha(out / "inputs.json"), weight=SHA256,
         reference_rows=len(ref), head_rows=len(head), query_only_rows=len(queries), device="mps"))
    model.fit(pd.DataFrame(x[ref],dtype=object), y[ref])
    save(area / "fitted.json", dict(launch=sha(area / "launch.json"), foundation_context_fits=1, parameter_updates=0))
    embeddings, probs = [], []
    for i in range(0,len(queries),192):
        features, pred = extract(model, queries[i:i+192])
        embeddings.append(features)
        probs.append(pred)
    z = np.concatenate(embeddings)
    p = np.concatenate(probs)
    ix = np.array([len(queries)-1, 123, 0, 653, len(head), 534, 1])
    repeated, _ = extract(model, queries[ix])
    require(np.isfinite(z).all() and np.isfinite(p).all() and np.allclose(p.sum(1),1.,atol=1e-5),
            "Invalid frozen features or class probabilities")
    shift = float(np.max(np.abs(z[ix]-repeated)))
    np.testing.assert_allclose(z[ix], repeated, rtol=2e-4, atol=2e-4)
    files = {}
    for name, value in (("embedding",z), ("head_indices",head), ("reference_probabilities",p)):
        path = area / (name + ".npy")
        np.save(path,value,allow_pickle=False)
        files[name] = sha(path)
    save(area / "sealed.json", dict(fit=sha(area / "fitted.json"), files=files, classes=model.classes_.tolist(),
         hidden_width=z.shape[1], reordered_row_count=len(ix), largest_reordered_difference=shift, api_requests=0))
    print("frozen vectors",z.shape,shift,flush=True)


def train(out):
    info = read(out / "inputs.json")
    data = bound(out,"fitting.joblib",info)
    x, y = _clean_categories(data["x"]), np.asarray(data["y"])
    classes = np.unique(y)
    panels = [_clean_categories(bound(out,r+"_features.joblib",info)) for r in ROLES]
    splits = np.cumsum([len(a) for a in panels])[:-1]
    area = out / "frozen_row_representations"
    stamped = read(area / "sealed.json")
    require(stamped["classes"] == classes.tolist(), "Changed class domain")
    for name, h in stamped["files"].items():
        require(sha(area / (name + ".npy")) == h, "Modified frozen features")
    head = np.load(area / "head_indices.npy", allow_pickle=False)
    z = np.load(area / "embedding.npy", allow_pickle=False)
    p = np.load(area / "reference_probabilities.npy", allow_pickle=False)
    q = np.concatenate(panels)
    onehot = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    tx, qx = onehot.fit_transform(x[head]), onehot.transform(q)
    scale = StandardScaler()
    a, b = scale.fit_transform(z[:len(head)]), scale.transform(z[len(head):])
    with threadpool_limits(limits=2):
        proj = PCA(n_components=min(32,a.shape[1]), svd_solver="randomized", random_state=173)
        at, bt = proj.fit_transform(a), proj.transform(b)
    def store(name, probabilities, fits):
        require(probabilities.shape == (len(q),len(classes)) and np.isfinite(probabilities).all()
                and np.allclose(probabilities.sum(1),1.,atol=1e-5), "Invalid head predictions")
        folder = out / ("rowembed_" + name)
        folder.mkdir(exist_ok=False)
        stamps = {}
        for role, mat in zip(ROLES, np.split(probabilities,splits), strict=True):
            path = folder / (role + ".npy")
            np.save(path,mat,allow_pickle=False)
            stamps[role] = sha(path)
        save(folder / "sealed.json", dict(protocol=sha(PLAN), encoder=sha(area / "sealed.json"), classes=classes.tolist(),
             predictions=stamps, head_fits=fits, api_requests=0))
    store("reference",p[len(head):],0)
    for name, extra in (("raw",False),("augmented",True)):
        u, v = (np.column_stack((tx,a)), np.column_stack((qx,b))) if extra else (tx,qx)
        with threadpool_limits(limits=2):
            clf = LogisticRegression(C=.1, solver="lbfgs", max_iter=1000).fit(u,y[head])
            pred = clf.predict_proba(v)
        store(name+"_linear",pred,1)
        tree = _tabular_catboost(y[head],6,3.)
        tree.set_params(thread_count=3)
        u,v = pd.DataFrame(x[head],dtype=object),pd.DataFrame(q,dtype=object)
        if extra:
            u = pd.concat([u,pd.DataFrame(at)],axis=1,ignore_index=True)
            v = pd.concat([v,pd.DataFrame(bt)],axis=1,ignore_index=True)
        tree.fit(u,y[head],cat_features=list(range(x.shape[1])))
        store(name+"_tree",tree.predict_proba(v),1)
    teacher = out / "distill_teachers"
    aligned = out / "adaptive_memory_components"
    receipts = read(aligned / "teachers.json")
    require(receipts["borrowed"] == sha(teacher / "sealed.json")
            and read(teacher / "sealed.json")["inputs"] == sha(out / "inputs.json")
            and read(teacher / "sealed.json")["predictions"] == sha(teacher / "out_of_fold.npy")
            and receipts["probabilities"] == sha(aligned / "aligned_out_of_fold.npy"), "Unbound held-out teachers")
    oof = (np.load(teacher / "out_of_fold.npy",allow_pickle=False)[:,1]
           + np.load(aligned / "aligned_out_of_fold.npy",allow_pickle=False)) / 2
    prior = np.concatenate([(cached_probability(out,"trial_attribute_alignment",r,classes)
                             + cached_probability(out,"trial_pretrained_icl",r,classes))/2 for r in ROLES])
    with threadpool_limits(limits=2):
        coef, _ = fit_offset(a.astype(float), (y[head] == classes[1]).astype(int), oof[head], positive=False)
    prob = expit(logit(np.clip(prior,.001,.999))+np.clip(b @ coef, -1.25, 1.25))
    store("offset",np.column_stack((1-prob,prob)),1)
    save(area / "heads.json", dict(encoder=sha(area / "sealed.json"), supervised_heads=5, unsupervised_pca=1,
         one_hot_vocabulary=1, feature_scaler=1, reused_oof_teacher_fits=12))


def grade(out):
    elect(out,METHODS,"rowembed",PLAN)
    info = read(out / "inputs.json")
    detail = {}
    for role in ROLES:
        y = bound(out,role+"_answers.joblib",info)
        detail[role] = {}
        for method in ("reference","raw_linear","augmented_linear","raw_tree","augmented_tree","offset"):
            area = out / ("rowembed_"+method)
            seal = read(area / "sealed.json")
            path = area / (role+".npy")
            require(sha(path)==seal["predictions"][role], "Changed diagnostic head")
            hard = np.asarray(seal["classes"])[np.load(path,allow_pickle=False).argmax(1)]
            detail[role][method] = int(np.sum(hard == y))
    save(out / "rowembed_election/standalone.json", detail)
    print(detail, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("encode","train","grade"))
    parser.add_argument("--out",required=True,type=lambda p:Path(p).resolve())
    parser.add_argument("--weight",type=lambda p:Path(p).resolve())
    args = parser.parse_args()
    if args.action == "encode":
        prepare(args.out,args.weight)
    else:
        (train if args.action == "train" else grade)(args.out)
