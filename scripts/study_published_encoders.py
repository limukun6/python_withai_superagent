"""Compare extra public encoders with a sealed portfolio on training-only group splits."""
import argparse
import hashlib
import os
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from scripts.compare_regrouped_portfolios import lock
from scripts.confirm_unused_training_groups import measure
from scripts.run_planner_self_review import source_identity
from scripts.study_portfolio_recruitment import score, solve
from scripts.study_resource_audition import ROLES, eligible, paired
from scripts.study_resource_menu import inventory
from scripts.unified_released_study import ROOT, env, read, require, sha
from superagent.ab_methods import PREFERENCE_CATALOG_ENV
from superagent.data import TrainingTask
from superagent.external_assets import activate_resource_directory
from superagent.local_sentence import bank_path, describe, encode
from superagent.metrics import MetricSpec
from superagent.paired_encoder import text_key
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups, take


PLAN = ROOT / "research/paired_encoder_search_1127/published_sibling_protocol.json"


def resources(root):
    directory = root / "sibling-library"
    previous = inventory(root / "potion-library")
    expanded = inventory(directory)
    published = read(root / "sibling-installation.json")
    registered = read(PLAN)["public_resources"]
    require(set(published) == set(registered), "Resource set differs from registration")
    extra = {}
    for repo, revision in registered.items():
        spec = describe(directory / repo.split("/")[-1])
        rec = published[repo]
        require(rec["revision"] == revision and rec["files"] == spec["files"], "Unmatched published bytes")
        extra[spec["signature"]] = (repo, spec)
    require(expanded == {**previous, **{k: v[1] for k, v in extra.items()}}
            and not previous.keys() & extra.keys(), "Resource change includes other encoders")
    activate_resource_directory(directory)
    return expanded, extra


def link_reference_banks(source, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for file in source.glob("*.npz"):
        target = destination / file.name
        if not target.exists():
            target.symlink_to(file.resolve())


def write_vectors(path, keys, vectors):
    temp = path.with_suffix(".pending")
    with temp.open("wb") as handle:
        np.savez(handle, keys=keys, vectors=vectors)
    temp.replace(path)


def cache(root, stage):
    expanded, added = resources(root)
    out = root / "published-siblings"
    if stage == "audition":
        payload = joblib.load(root / "resource-audition/worker_input.joblib")
        rows = list(payload["X_train"]) + list(payload["X_search"])
        destination = out / "audition" / "fit_features"
        reference = root / "arctic-menu/fit_features"
    else:
        payload = joblib.load(root / "portfolio-recruitment/comparison/challenger/run/agent/refit_data.joblib")
        rows = payload["X_train"]
        destination = out / "ordinary/comparison/fit_features"
        reference = root / "portfolio-recruitment/comparison/fit_features"
    link_reference_banks(reference, destination)
    texts = sorted({s for pair in rows for s in pair})
    keys = [text_key(s) for s in texts]
    require(len(keys) == len(set(keys)), "Hash collision in training texts")
    for signature, (name, spec) in sorted(added.items()):
        target = bank_path(destination, spec)
        parts = out / stage / "vector_parts" / signature
        parts.mkdir(parents=True, exist_ok=True)
        allowed = set(keys)
        previous = {}
        prior = bank_path(out / "audition/fit_features", spec)
        if stage == "ordinary" and prior.exists():
            with np.load(prior, allow_pickle=False) as saved:
                previous = {k: v for k, v in zip(saved["keys"].tolist(), saved["vectors"], strict=True) if k in allowed}
        if not target.exists():
            start = time.monotonic()
            for i in range(0, len(keys), 512):
                subkeys = keys[i:i+512]
                part = parts / f"{i:06d}.npz"
                if part.exists():
                    continue
                missing = [(k, t) for k, t in zip(subkeys, texts[i:i+512], strict=True) if k not in previous]
                values = dict(previous)
                if missing:
                    values.update(zip((k for k, _ in missing), encode(spec, [t for _, t in missing]), strict=True))
                write_vectors(part, subkeys, np.stack([values[k] for k in subkeys]))
                print(stage, name, min(i+512, len(keys)), "/", len(keys), flush=True)
            vectors = []
            for i in range(0, len(keys), 512):
                with np.load(parts / f"{i:06d}.npz", allow_pickle=False) as chunk:
                    require(chunk["keys"].tolist() == keys[i:i+512], "A vector checkpoint has different rows")
                    vectors.append(chunk["vectors"])
            matrix = np.concatenate(vectors)
            require(matrix.shape == (len(keys), spec["width"]) and np.isfinite(matrix).all(), "Invalid vectors")
            write_vectors(target, keys, matrix)
            lock(parts / "receipt.json", dict(rows=len(keys), frozen_inferred=len(keys)-len(previous),
                 reused=len(previous), seconds=round(time.monotonic()-start, 2), bank=sha(target)))
        with np.load(target, allow_pickle=False) as saved:
            require(saved["keys"].tolist() == keys, "Additional rows in the frozen cache")
    print(stage, len(expanded), "discoverable encoders", len(texts), "allowed texts", flush=True)


def audition(root):
    _, extra = resources(root)
    out = root / "published-siblings/audition"
    original = root / "resource-audition"
    reference = read(root / "static-audition/inputs.json")
    weights = reference["reference"]
    require(np.isclose(sum(weights.values()), 1) and min(weights.values()) > 0,
            "Invalid elected portfolio")
    data_file = root / "arctic-menu/original/run/agent/refit_data.joblib"
    require(sha(data_file) == reference["source"], "Training source changed")
    payload = original / "worker_input.joblib"
    registry = root / "analogy-portfolio-raw/portfolio_revised/run/agent/runtime_registry_0.json"
    fit = read(original / "fitting.json")
    require(sha(payload) == fit["payload"] == reference["input"] and sha(registry) == fit["registry"],
            "Components would use different training inputs")
    task = joblib.load(data_file)
    metric = MetricSpec.from_payload(task)
    rows, y = task["X_train"], np.asarray(task["y_train"])
    classes = np.asarray(metric.labels)
    published = {eligible(k): name for k, (name, _) in extra.items()}
    require(metric.name == "accuracy" and not (published.keys() & weights.keys()), "Unmatched comparison")
    # The four pre-existing models and the two challengers see the same fitting rows.
    groups = sample_groups(TrainingTask(task["kind"], rows, y), "shared_text")
    info = read(original / "inputs.json")
    folds = [h for _, h in StratifiedGroupKFold(5, shuffle=True, random_state=info["seed"]).split(groups,y,groups)]
    panels = dict(zip(ROLES, folds[:3], strict=True), fitting=np.sort(np.concatenate(folds[3:])))
    require(all(fingerprint(ix) == info["partitions"][role]["key"] for role, ix in panels.items())
            and sum(len(set(groups[ix])) for ix in panels.values()) == len(set(groups)), "A group changed role")
    physical = out / "physical.jsonl"
    settings = env(root / "sibling-library", physical)
    activate_resource_directory(root / "sibling-library", settings)
    settings.update({PREFERENCE_CATALOG_ENV: "all-installed", "SUPERAGENT_PAIRED_FEATURE_CACHE": str(out/"fit_features")})
    os.environ.update(SUPERAGENT_PAIRED_FEATURE_CACHE=str(out/"fit_features"), HF_HUB_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1")
    lock(out/"inputs.json", dict(protocol=sha(PLAN), runtime=source_identity(ROOT), published=published,
         source=sha(data_file), payload=fit["payload"], registry=fit["registry"], reference=weights,
         partitions=info["partitions"], installation=sha(root/"sibling-installation.json")))
    components = read(original/"predictions.json")["components"]
    def probabilities(role, requested):
        result = {}
        for name in requested:
            folder = (original if name in weights else out) / name
            if role == "nomination":
                p = np.asarray(measure(name, payload, registry, folder, settings, metric, y[panels[role]]))
            else:
                model = joblib.load(folder/"trial.joblib")["model"]
                columns = [list(model.classes_).index(c) for c in classes]
                p = model.predict_proba(take(rows, panels[role]))[:,columns]
            require(p.shape == (len(panels[role]),len(classes)) and np.isfinite(p).all()
                    and np.allclose(p.sum(1),1), "Invalid probabilities")
            if name in weights:
                require(hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest() == components[name][role],
                        "Original probabilities changed")
            result[name] = p
        return result

    def evaluate(role, proposals):
        requested = sorted(set(weights) | {m for group, _ in proposals for m in group})
        bank = probabilities(role, requested)
        base = sum(w*bank[m] for m,w in weights.items())
        truth = y[panels[role]]
        right = classes[base.argmax(1)] == truth
        scores = {}
        for group, share in proposals:
            p = base if not group else (1-share)*base + share*sum(bank[m] for m in group)/len(group)
            scores[group,share] = paired(classes[p.argmax(1)] == truth, right)
        return int(right.sum()), scores

    names = sorted(published)
    proposals = [((),0.)] + [(group,w) for group in [(names[0],),(names[1],),tuple(names)] for w in (.25,.5,1.)]
    base, measured = evaluate("nomination", proposals)
    choice = min(measured, key=lambda opt:(-measured[opt]["correct"],opt[1],len(opt[0]),opt[0]))
    group, share = choice
    lock(out/"nomination.json", dict(inputs=sha(out/"inputs.json"),rows=len(panels["nomination"]),
         protected=base,selected=list(group),share=share,
         options=[dict(names=list(g),share=w,**m) for (g,w),m in measured.items()]))
    checks = {}
    advance = bool(share)
    for role in ROLES[1:]:
        if not advance:
            break
        base_count, observed = evaluate(role,[choice])
        checks[role] = dict(protected=base_count,rows=len(panels[role]),**observed[choice])
        lock(out/(role+".json"),checks[role])
        advance = observed[choice]["net"] > 0
    count = len(physical.read_text().splitlines())
    require(count == len(extra), "Unexpected actual fits")
    result = dict(nomination=sha(out/"nomination.json"),published=published,selected=list(group),share=share,
        protected_correct=base,initial_rows=len(panels["nomination"]),checks=checks,advance=advance,
        supervised_fits=count,reused_classifier_fits=len(weights),model_api_requests=0,
        designated_validation_opened=False,official_test_opened=False,H2_opened=False)
    lock(out/"results.json",result)
    print(result,flush=True)


def ordinary(root):
    expanded, added = resources(root)
    prior = root / "static-menu-cold"
    out = root / "published-siblings/ordinary"
    case = out / "comparison"
    case.mkdir(parents=True,exist_ok=True)
    for name in ("p.zip","q.zip","truth.joblib"):
        target = case / name
        if not target.exists():
            target.symlink_to(prior/"comparison"/name)
    if not (case/"control").exists():
        (case/"control").symlink_to(prior/"comparison/challenger")
    previous = read(prior/"inputs.json")
    require(sorted(set(expanded)-set(added)) == previous["resources"] and
            previous["challenger"] == source_identity(ROOT),"Runtime or old resource set changed")
    lock(out/"inputs.json",dict(previous,plan=sha(PLAN),control=previous["challenger"],resources=sorted(expanded),
        candidates_added=sorted(added),prior=sha(case/"control/sealed.json")))
    if not (case/"challenger/sealed.json").exists():
        solve(root, ROOT, "comparison", "challenger", out=out, library=root/"sibling-library")
    score(root, "comparison", out=out)
    agent = case/"challenger/run/agent"
    history, steps = read(agent/"history.json"), read(agent/"decisions.json")
    final = read(agent/"run_outcome.json")
    new_names = {eligible(k): v[0] for k,v in added.items()}
    old_steps = read(case/"control/run/agent/decisions.json")
    def menu(actions):
        return {m for a in actions[0]["alternatives"] for m in a["members"]}
    require(menu(steps)-menu(old_steps) == set(new_names) and menu(old_steps) <= menu(steps),
            "Unrelated initial menu change")
    observations = {}
    for n,repo in new_names.items():
        offered = [d["step"] for d in steps if any(a["action"]=="explore" and a["members"]==[n] for a in d["alternatives"])]
        tested = [dict(split=h.get("split_id"),status=h["status"]) for h in history if h.get("candidate")==n]
        observations[repo] = dict(candidate=n,offered_at=offered,trials=tested,elected=n in final["members"])
    result = dict(score=sha(case/"results.json"),seal=sha(case/"challenger/sealed.json"),resources=observations,
        initial_count=len(menu(steps)),previous_count=len(menu(old_steps)),final=final["members"],
        fits=final["attempted_classifier_fits"],model_api_requests=0)
    lock(out/"choice.json",result)
    print(result,flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("audition-cache","ordinary-cache","audition","ordinary"))
    parser.add_argument("--root",required=True,type=lambda s: Path(s).resolve())
    args=parser.parse_args()
    if args.action.endswith("-cache"):
        cache(args.root,args.action.split("-")[0])
    elif args.action=="audition":
        audition(args.root)
    else:
        ordinary(args.root)
