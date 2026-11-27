"""Adapt a frozen table model's final readout on disjoint out-of-fold contextual activations."""
import argparse
import gc
from importlib.metadata import version
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from scripts.study_task1_adaptive_memory import cached_probability
from scripts.study_task1_candidates import ROLES, bound
from scripts.study_task1_cross_context import fit_offset
from scripts.study_task1_tabm import elect
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.models import _clean_categories
from superagent.splits import sample_groups
from superagent.table_foundation import SHA256

PLAN = ROOT / "research/task1_feedback_extension_1127/adapt_foundation_readout.json"
METHODS = ("frozen", "calibrated", "adapted", "incumbent")


def extract(clf, x):
    layer = clf.model_.icl_predictor.decoder[-1]
    hidden = []
    def input_state(module, args):
        hidden.append(args[0].detach().float().cpu().numpy().copy())
    handle = layer.register_forward_pre_hook(input_state)
    try:
        published = clf.predict_proba(pd.DataFrame(x,dtype=object))[:,1]
    finally:
        handle.remove()
    require(len(hidden) == 1 and hidden[0].shape[:2] == (1,len(x)), "Unexpected decoder input shape")
    matrix = hidden[0][0]
    w = layer.weight.detach().float().cpu().numpy()
    b = layer.bias.detach().float().cpu().numpy()
    rebuilt = expit((matrix @ (w[1]-w[0]) + b[1]-b[0])/clf.softmax_temperature)
    np.testing.assert_allclose(rebuilt,published,rtol=2e-4,atol=2e-4)
    return matrix,published


def run(out,weight):
    import torch
    from tabicl import TabICLClassifier
    require(version("tabicl") == "2.0.3" and sha(weight) == SHA256, "Encoder identity mismatch")
    torch.set_num_threads(2)
    info = read(out / "inputs.json")
    fitting = bound(out,"fitting.joblib",info)
    x = _clean_categories(fitting["x"])
    classes, y = np.unique(fitting["y"],return_inverse=True)
    panels = [_clean_categories(bound(out,r+"_features.joblib",info)) for r in ROLES]
    q = np.concatenate(panels)
    ref = out / "adapted_readout_contexts"
    ref.mkdir(exist_ok=False)
    save(ref / "launch.json",dict(protocol=sha(PLAN),inputs=sha(out / "inputs.json"),weight=SHA256,rows=len(x),queries=len(q)))
    groups = sample_groups(SimpleNamespace(kind="categorical_tabular",X_train=x))
    covered = np.zeros(len(y),dtype=int)
    oof, raw_oof = None, np.zeros(len(y))
    qstates, raw_query = [], []
    for k,(fit,held) in enumerate(StratifiedGroupKFold(4,shuffle=True,random_state=72731).split(x,y,groups)):
        require(set(groups[fit]).isdisjoint(groups[held]), "Repeated group in own reference")
        model = TabICLClassifier(n_estimators=1, norm_methods="none", feat_shuffle_method="none", class_shuffle_method="none",
                batch_size=1, kv_cache=True, model_path=weight,allow_auto_download=False,device="mps",
                use_amp=False,use_fa3=False,n_jobs=1,random_state=173)
        model.fit(pd.DataFrame(x[fit],dtype=object),y[fit])
        save(ref / f"fitted_{k}.json",dict(launch=sha(ref / "launch.json"),training=len(fit),held=len(held),context_fits=1))
        both = np.concatenate((x[held],q))
        states, published = [], []
        for i in range(0,len(both),192):
            z,p = extract(model,both[i:i+192])
            states.append(z)
            published.append(p)
        z,p = np.concatenate(states),np.concatenate(published)
        if oof is None:
            oof = np.empty((len(y),z.shape[1]),dtype=z.dtype)
        oof[held],raw_oof[held] = z[:len(held)],p[:len(held)]
        covered[held] += 1
        qstates.append(z[len(held):])
        raw_query.append(p[len(held):])
        del model
        gc.collect()
        print("reference",k,len(fit),z.shape[1],flush=True)
    require(np.all(covered == 1) and np.isfinite(oof).all(), "Incomplete contextual out-of-fold representation")
    with threadpool_limits(limits=2):
        pca = PCA(n_components=32,whiten=True,svd_solver="randomized",random_state=173)
        a = pca.fit_transform(oof).astype(float)
        b = np.stack([pca.transform(z) for z in qstates])
    raw_query = np.asarray(raw_query)
    old = out / "distill_teachers"
    aligned = out / "adaptive_memory_components"
    linked = read(aligned / "teachers.json")
    require(linked["borrowed"] == sha(old/"sealed.json") and read(old/"sealed.json")["inputs"] == sha(out/"inputs.json")
            and linked["probabilities"] == sha(aligned/"aligned_out_of_fold.npy")
            and read(old/"sealed.json")["predictions"] == sha(old/"out_of_fold.npy"), "Unbound existing teachers")
    prior = (np.load(old/"out_of_fold.npy",allow_pickle=False)[:,1]+np.load(aligned/"aligned_out_of_fold.npy",allow_pickle=False))/2
    prior_q = np.concatenate([(cached_probability(out,"trial_attribute_alignment",r,classes)+
                                cached_probability(out,"trial_pretrained_icl",r,classes))/2 for r in ROLES])
    results = {"frozen":raw_query.mean(axis=0)}
    def odds(p):
        return logit(np.clip(p,.001,.999))
    for name,base,query_base in (("calibrated",raw_oof,raw_query),("adapted",raw_oof,raw_query),("incumbent",prior,prior_q)):
        train_design = np.column_stack((np.ones(len(y)),odds(raw_oof))) if name=="calibrated" else a
        query_design = np.stack((np.ones_like(raw_query),odds(raw_query)),axis=-1) if name=="calibrated" else b
        with threadpool_limits(limits=2):
            coef,stats = fit_offset(train_design,y,base,positive=False)
        results[name] = expit(odds(query_base)+np.clip(query_design@coef,-1.25,1.25)).mean(axis=0)
        save(ref / (name + ".json"),dict(supervised_readout_fits=1,stats=stats))
    for name,values in results.items():
        require(values.shape==(len(q),) and np.isfinite(values).all(), "Incomplete adapted readout")
        folder = out / ("adaptread_"+name)
        folder.mkdir(exist_ok=False)
        stamps = {}
        for role,p in zip(ROLES,np.split(values,np.cumsum([len(x) for x in panels])[:-1]),strict=True):
            dest = folder/(role+".npy")
            np.save(dest,np.column_stack((1-p,p)),allow_pickle=False)
            stamps[role]=sha(dest)
        save(folder/"sealed.json",dict(protocol=sha(PLAN),classes=classes.tolist(),predictions=stamps,api_requests=0))
    save(ref/"sealed.json",dict(launch=sha(ref/"launch.json"),new_contexts=4,adapted_heads=3,pca_fits=1,
                                hidden_width=oof.shape[1],zero_overlap=True,api_requests=0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("run","grade"))
    parser.add_argument("--out",required=True,type=lambda p:Path(p).resolve())
    parser.add_argument("--weight",type=lambda p:Path(p).resolve())
    args=parser.parse_args()
    if args.action=="run":
        run(args.out,args.weight)
    else:
        elect(args.out,METHODS,"adaptread",PLAN)
