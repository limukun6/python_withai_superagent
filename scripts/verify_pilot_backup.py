"""Rebuild the two independent confirmation stages from frozen probabilities and original training rows."""
import argparse
from pathlib import Path
import tempfile
from unittest.mock import patch
import zipfile

import numpy as np

from scripts.evaluate_pilot_nomination import project
from scripts.run_conditional_handoff import load_export
from scripts.study_crossfit_challenger import prepare as reconstruct_packets, study_splits
from scripts.study_pilot_backup import score
from scripts.unified_released_study import ROOT, read, require, save, sha
from scripts.verify_crossfit_challenger import checksums, seal_directory
from scripts.verify_lifecycle_audit import copy
from scripts.verify_pilot_nomination import PUBLIC as PARENT, receipts, verify_one_receipt
from scripts.verify_unified_raw import MEMBERS
from superagent.data import load_training_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups
from tools.challenger_training import rows_at, ties
from tools.pilot_challenger import (BACKUP_STRATEGIES, advance, approve, backup_roles, compare_vote,
                                    panel_scores, pilot_roles, shortlist)

PUBLIC = ROOT / "research/pilot_backup_1124"


def publish(out, smoke):
    for f in ("protocol.json","frozen.json","result.json"):
        copy(out/f, PUBLIC/f)
    copy(smoke/"smoke.json",PUBLIC/"smoke/smoke.json")
    for s in BACKUP_STRATEGIES:
        for f in ("run/pilot_protocol.json","run/nomination.json","run/first_check.json","run/decision.json",
                  "run/pilot_completion.json","direct/predictions.csv","cached/predictions.csv"):
            copy(smoke/s/f, PUBLIC/"smoke"/s/f)
    save(PUBLIC/"smoke/fit_receipts.json",receipts(smoke,BACKUP_STRATEGIES))
    for i in range(5):
        src, dst = out/f"fold_{i}", PUBLIC/f"fold_{i}"
        for f in ("sealed.json", *read(src/"sealed.json")["records"]):
            copy(src/f,dst/f)
        save(dst/"fit_receipts.json",receipts(src,BACKUP_STRATEGIES))
    seal_directory(PUBLIC)


def stage_score(scores, key, ids, n, task, groups):
    p, recorded = scores[key], scores[key+"_ids"]
    if not n:
        require(not p.size and not recorded.size,"Unnominated model was evaluated")
        return None
    require(np.array_equal(ids,recorded) and p.shape==(len(ids),3) and np.isfinite(p).all()
            and np.all((p>=0)&(p<=1)), "Invalid confirmation groups or probabilities")
    return compare_vote(p[:,:2].mean(1),p[:,2],ties(rows_at(task,ids)),task.y_train[ids],groups[ids])


def verify_fold(folder, parent, packet, outer):
    task = load_training_task(packet)
    package = sha(packet)
    groups = sample_groups(task,"shared_text")
    roles = backup_roles(pilot_roles(groups,task.y_train),groups,task.y_train)
    desc = {k:dict(ids=fingerprint(ix.tolist()),rows=len(ix),groups=len(set(groups[ix]))) for k,ix in roles.items()}
    require(sum(v["groups"] for v in desc.values())==len(set(groups)),"Confirmation group collision")
    parent_receipts = read(parent/"fit_receipts.json")
    available = set()
    for payload in parent_receipts.values():
        r = payload["fit"]
        available.add((r["identity"]["ids"],r["name"],r["model"],r["ledger"]))
    recs = read(folder/"fit_receipts.json")
    seal = read(folder/"sealed.json")
    require(sha(parent/"sealed.json")==seal["parent"],"Wrong parent fold")
    expect_keys, chosen, refit_hashes = set(), {}, {}
    for s in BACKUP_STRATEGIES:
        run=folder/s/"run"
        proto,nom,first,decision,end,selection=(read(run/f) for f in ("pilot_protocol.json","nomination.json",
            "first_check.json","decision.json","pilot_completion.json","agent/final_selection.json"))
        names=proto["inventory"]["candidates"]
        require(proto["package"]==package and proto["roles"]==desc and
                proto["strategy"]==nom["strategy"]==end["strategy"]==s and nom["members"]==names
                and nom["learning_rows"]==desc["learn"]["ids"],"Inputs or goal differ")
        with np.load(run/"feedback.npz",allow_pickle=False) as fb:
            learn=roles["learn"]
            require(np.array_equal(fb["learn_ids"],learn),"Nomination rows changed")
            observed=panel_scores(names,fb["learn"],rows_at(task,learn),task.y_train[learn],groups[learn])
            a,b=shortlist(observed)
            require((a,b)==(nom["chosen"],nom["backup"]) and observed==nom["learning"],"Nomination pair differs")
            check=stage_score(fb,"first_check",roles["first_check"],a,task,groups)
            candidate=advance(a,b,check,s)
            expected_stage=dict(first=a,backup=b,first_check=check,advanced=candidate,nomination=sha(run/"nomination.json"))
            require(first==expected_stage,"First confirmation changed or finalist changed")
            final=stage_score(fb,"confirm",roles["confirm"],candidate,task,groups)
        expected=approve(candidate,observed,final,len(roles["confirm"]))
        expected.update(nomination=sha(run/"nomination.json"),feedback=sha(run/"feedback.npz"),
                        first_check_sha256=sha(run/"first_check.json"))
        chosen[s]=expected["approved"] or "baseline"
        used=names[:2]+([expected["approved"]] if expected["approved"] else [])
        require(expected==decision and end["decision_sha256"]==sha(run/"decision.json")
                and selection["selected_experiment"]==chosen[s] and selection["ensemble_components"]==used
                and selection["model_sha256"]==end["model_sha256"] and not selection["test_accessed"],
                "Final decision or exported members differ")
        ordered=[("pilot",n) for n in names]+[("refit",n) for n in used]
        require(ordered==[(r["stage"],r["candidate"]) for r in end["actions"]],"Unaccounted fit")
        prev=set(available)
        for a in end["actions"]:
            stage,n=a["stage"],a["candidate"]
            key=f"{s}/{stage}/{n}"
            expect_keys.add(key)
            ids=desc["pilot"]["ids"] if stage=="pilot" else fingerprint(list(range(len(groups))))
            identity=dict(ids=ids,package=package)
            r=verify_one_receipt(recs[key],identity,n)
            entry=(ids,n,r["model"],r["ledger"])
            require(a["identity"]==identity and a["model"]==r["model"] and a["ledger"]==r["ledger"] and
                    (entry in prev if a["reused"] else stage=="refit" and n not in names[:2]),
                    "Reuse or new fit differs from the registered budget")
            available.add(entry)
            if stage=="refit":
                require(refit_hashes.setdefault(n,r["model"])==r["model"],"Chosen heads differ between arms")
        cost=seal["costs"][s]
        require(cost["new"]==end["new_classifier_fits"]==sum(not a["reused"] for a in end["actions"])
                and cost["reused"]==end["reused_classifier_fits"] and cost["cold"]==len(ordered)==cost["new"]+cost["reused"]<=9
                and end["model_api_calls"]==0, "Fit budget or API count differs")
    require(refit_hashes==seal["model_hashes"] and expect_keys==set(recs) and
            sum(c["new"] for c in seal["costs"].values())<=1, "Unexpected comparison fit or component")
    with np.load(folder/"refit.npz",allow_pickle=False) as p:
        votes=project(seal["refit_members"],p["scores"],outer)
    recorded=load_export(folder/"rows.csv")
    previous=load_export(parent/"rows.csv")
    require(np.array_equal(recorded["row_id"],np.arange(len(outer))) and
            np.array_equal(votes["baseline"],recorded["baseline"]) and np.array_equal(previous["baseline"],recorded["baseline"]) and
            all(np.array_equal(previous[s+"_approved"],recorded[s]) for s in ("pilot-accuracy","pilot-brier")), "Mismatched controls")
    for s in BACKUP_STRATEGIES:
        native=load_export(folder/s/"direct/predictions.csv")
        require(np.array_equal(native["row_id"],recorded["row_id"]) and np.array_equal(native["prediction"],votes[chosen[s]])
                and np.array_equal(native["prediction"],recorded[s]),"Independent native export differs")


def verify(archive):
    require(checksums(PUBLIC)==(PUBLIC/"SHA256SUMS").read_text() and checksums(PARENT)==(PARENT/"SHA256SUMS").read_text(),
            "Published evidence changed")
    opened=set()
    original=zipfile.ZipFile.open
    def guard(z,member,*args,**kwargs):
        if isinstance(z.filename,(str,Path)) and Path(z.filename).resolve()==archive.resolve():
            name=member.filename if isinstance(member,zipfile.ZipInfo) else member
            require(name in MEMBERS["titles"],"Unexpected course archive member")
            opened.add(name)
        return original(z,member,*args,**kwargs)
    with patch.object(zipfile.ZipFile,"open",guard),tempfile.TemporaryDirectory() as tmp:
        packets=Path(tmp)/"packets"
        reconstruct_packets(archive,packets)
        task,_,splits=study_splits(archive)
        for i,(_,held) in enumerate(splits):
            key=f"fold_{i}"
            verify_fold(PUBLIC/key,PARENT/key,packets/key/"train.zip",rows_at(task,held))
        score(PUBLIC,PARENT,archive)
    require(opened==set(MEMBERS["titles"]),"Missing course training file audit")
    smoke=PUBLIC/"smoke"
    srow=read(smoke/"smoke.json")
    require(srow["parent"]==sha(PARENT/"smoke/smoke.json") and srow["model_api_calls"]==0, "Unmatched anonymous input")
    for s,rec in srow["results"].items():
        require(sha(smoke/s/"direct/predictions.csv")==sha(smoke/s/"cached/predictions.csv")==rec["predictions"]
                and rec["decision"]==sha(smoke/s/"run/decision.json") and rec["model"]==read(smoke/s/"run/pilot_completion.json")["model_sha256"],
                "Anonymous delivery differs")
    print("Verified nominations, two disjoint checks, physical fits and all ten native backup exports.")


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    subs=p.add_subparsers(dest="stage",required=True)
    pub=subs.add_parser("publish")
    pub.add_argument("out",type=Path)
    pub.add_argument("smoke",type=Path)
    verify_cmd=subs.add_parser("verify")
    verify_cmd.add_argument("archive",type=Path)
    a=p.parse_args()
    if a.stage=="publish":
        publish(a.out,a.smoke)
    else:
        verify(a.archive)
