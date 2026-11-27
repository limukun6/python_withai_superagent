"""Give each discovered encoder a matched supervised trial on separated training roles."""
import argparse
import hashlib
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.compare_regrouped_portfolios import lock
from scripts.confirm_unused_training_groups import measure
from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT, env, read, require, sha
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.data import TrainingTask
from superagent.external_assets import activate_resource_directory
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups, take


PLAN = ROOT / "research/paired_encoder_search_1127/resource_audition_protocol.json"
ROLES = ("nomination", "confirmation", "audit")
SHARES = (.25, .5, 1.)


def eligible(signature):
    return "local_sentence_" + signature[:20]


def rights(bank, baseline, labels, classes, member=None, share=0.):
    mix = baseline if member is None else (1. - share) * baseline + share * bank[member]
    return classes[mix.argmax(axis=1)] == labels


def paired(candidate, original):
    fixed = int(np.count_nonzero(candidate & ~original))
    lost = int(np.count_nonzero(original & ~candidate))
    return dict(correct=int(candidate.sum()), fixed=fixed, lost=lost, net=fixed-lost)


def run(root):
    experiment = root / "arctic-menu"
    source = experiment / "original/run/agent/refit_data.joblib"
    menus = read(experiment / "inputs.json")
    election = root / "analogy-portfolio-raw/portfolio_revised"
    final = read(election / "run/agent/final_selection.json")
    require(final["model_sha256"] == read(election / "sealed.json")["model_sha256"],
            "Different protected election")
    data = joblib.load(source)
    metric = MetricSpec.from_payload(data)
    rows, truth = data["X_train"], np.asarray(data["y_train"])
    classes = np.asarray(metric.labels)
    require(len(truth)==menus["fit_rows"] and len(set(truth))==2 and metric.name=="accuracy",
            "Unexpected internal sample")
    groups = sample_groups(TrainingTask(data["kind"], rows, truth), "shared_text")
    seed = int(hashlib.sha256(b"AIAA resource audition separated roles v1").hexdigest()[:8],16)
    splitter = StratifiedGroupKFold(n_splits=5, random_state=seed, shuffle=True)
    folds = [idx for _,idx in splitter.split(groups,truth,groups)]
    panels = dict(zip(ROLES,folds[:3]))
    fit = np.sort(np.concatenate(folds[3:]))
    group_sets = [set(groups[ix]) for ix in [fit,*panels.values()]]
    require(sum(map(len,group_sets))==len(set(groups)), "Shared-text groups cross roles")
    protected = dict(zip(final["ensemble_components"],final["weights"],strict=True))
    require(np.isclose(sum(protected.values()),1.),"Invalid protected weights")
    signatures = {eligible(s):s for s in menus["expanded_encoders"]}
    old = {eligible(s) for s in menus["original_encoders"]}
    added = eligible(menus["signature"])
    require(set(signatures)-old=={added}, "Inventory no longer differs by one resource")
    members = sorted(set(protected) | set(signatures))
    require(len(members)<=10, "Audition is over budget")
    out = root / "resource-audition"
    out.mkdir(exist_ok=True)
    lock(out/"inputs.json",dict(protocol=sha(PLAN), runtime=source_identity(ROOT), source=sha(source),
         menu=sha(experiment/"inputs.json"), election=sha(election/"sealed.json"), seed=seed,
         groups=len(set(groups)), partitions={role:dict(rows=len(ix),groups=len(set(groups[ix])),key=fingerprint(ix))
             for role,ix in dict(fitting=fit,**panels).items()},protected=protected,resources=signatures))
    library = root/"arctic-library"
    physical = out/"physical.jsonl"
    settings = env(library, physical)
    activate_resource_directory(library,settings)
    activate_resource_directory(library)
    settings.update({PREFERENCE_CATALOG_ENV:"all-installed","SUPERAGENT_PAIRED_FEATURE_CACHE":str(experiment/"fit_features")})
    os.environ.update({k:settings[k] for k in ("HF_HUB_OFFLINE","TRANSFORMERS_OFFLINE","SUPERAGENT_PAIRED_FEATURE_CACHE")})
    registry = election/"run/agent/runtime_registry_0.json"
    payload = out/"worker_input.joblib"
    first = panels["nomination"]
    if not payload.exists():
        joblib.dump(dict(kind=data["kind"], X_train=take(rows,fit), y_train=truth[fit],
            X_search=take(rows,first), y_search=truth[first], metric_spec=metric.record(),
            pair_semantics=data["pair_semantics"], capture_probabilities=True, retain_model=True),payload)
    lock(out/"fitting.json",dict(inputs=sha(out/"inputs.json"),payload=sha(payload),registry=sha(registry)))
    bank = {role:{} for role in ROLES}
    predictions = {}
    for member in members:
        folder = out/member
        bank["nomination"][member] = np.asarray(measure(member,payload,registry,folder,settings,metric,truth[first]))
        model = joblib.load(folder/"trial.joblib")["model"]
        class_columns = [list(model.classes_).index(c) for c in classes]
        for role in ROLES[1:]:
            matrix = model.predict_proba(take(rows,panels[role]))[:,class_columns]
            require(np.isfinite(matrix).all() and np.allclose(matrix.sum(axis=1),1), "Invalid model probabilities")
            bank[role][member] = matrix
        predictions[member] = {role:hashlib.sha256(np.ascontiguousarray(bank[role][member]).tobytes()).hexdigest()
                               for role in ROLES}
        print(member,flush=True)
    lock(out/"predictions.json",dict(fitting=sha(out/"fitting.json"),components=predictions))
    base = {role:sum(w*bank[role][m] for m,w in protected.items()) for role in ROLES}
    def outcome(role, member=None, share=0.):
        return rights(bank[role],base[role],truth[panels[role]],classes,member,share)
    nomination = {}
    for name,options in (("original",old),("expanded",set(signatures))):
        alternatives = [(None,0.),*((m,w) for m in options for w in SHARES)]
        member, share = min(alternatives,key=lambda a:(-int(outcome("nomination",*a).sum()),
                                                      a[1],signatures.get(a[0],"")))
        nomination[name] = dict(member=member,share=share,
            paired=paired(outcome("nomination",member,share),outcome("nomination")))
    lock(out/"nomination.json",dict(predictions=sha(out/"predictions.json"),choices=nomination))
    decisions = {}
    for name, choice in nomination.items():
        member,share = choice["member"],choice["share"]
        delta = paired(outcome("confirmation",member,share),outcome("confirmation"))
        use = member is not None and delta["net"]>0
        decisions[name] = dict(member=member if use else None,share=share if use else 0.,
                               confirmation=delta)
    lock(out/"confirmation.json",dict(nomination=sha(out/"nomination.json"),choices=decisions))
    audit = {name:outcome("audit",choice["member"],choice["share"]) for name,choice in decisions.items()}
    ordinary = outcome("audit")
    summaries = {role:dict(rows=len(panels[role]),protected_correct=int(outcome(role).sum()),
        standalone={m:int(outcome(role,m,1.).sum()) for m in members},
        added_blends={str(w):paired(outcome(role,added,w),outcome(role)) for w in SHARES}) for role in ROLES}
    total = len(physical.read_text().splitlines())
    require(total==len(members),"Unexpected number of measured classifier fits")
    changed = paired(audit["expanded"],audit["original"])
    result = dict(confirmation=sha(out/"confirmation.json"),summaries=summaries,decisions=decisions,
         audit_vs_protected={n:paired(v,ordinary) for n,v in audit.items()},expanded_vs_original=changed,
         warranted_transfer=(decisions["expanded"]["member"]==added and changed["net"]>0 and
                             int(audit["expanded"].sum())>int(ordinary.sum())),
         fits=total,model_api_requests=0,designated_validation_opened=False,other_outer_labels_opened=False,
         official_test_opened=False,H2_opened=False)
    lock(out/"results.json", result)
    print(decisions,changed,result["warranted_transfer"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True,type=lambda s:Path(s).resolve())
    run(parser.parse_args().root)
