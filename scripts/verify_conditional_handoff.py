"""Publish small routing receipts and independently score them from the course training split."""
import argparse
import csv
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np

from scripts.prepare_conditional_handoff import EVIDENCE, GATE, PLAN, PUBLIC_BASE, ROOT, partition
from scripts.run_conditional_handoff import STUDY, changes, comparison, load_export, score_panel
from scripts.study_local_handoff import PLAN as LOCAL_PLAN, POLICY_SOURCE, SCRIPT as LOCAL_STUDY, current_rule
from scripts.study_handoff_feedback_volume import PLAN as VOLUME_PLAN
from scripts.unified_released_study import read, require, save, sha, title_rows
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups
from tools.preference_handoff import inputs, predictions, select

PUBLIC=ROOT/"research/unified_handoff_1124"


def accounting(study):
    records=[]
    for tag in ("replay",*(f"fold_{i}" for i in range(5))):
        folder=study if tag=="replay" else study/tag
        fit_dir=study/"replay_fits" if tag=="replay" else folder/"gate_fits"
        local=study/"local_replay" if tag=="replay" else folder/"local"
        recorded={}
        for name in ("hard","regions","soft"):
            entry=read(fit_dir/(name+".json"))
            admission=fit_dir/(name+".admission.json")
            require(sha(admission)==entry["admission_sha256"] and read(admission)["maximum_supervised_fits"]==1,
                    "Gate receipt differs")
            recorded[name]=entry["policy"]["supervised_fits"]
        mem=read(local/"decision.json")
        require(sha(local/"memory.joblib")==mem["memory"],"Saved supervised memory differs")
        for role,key in (("memory","supervised_memory_fits"),("threshold","supervised_threshold_selections")):
            recorded[role]=mem[key]
            require(read(local/(role+".admission.json"))["maximum_supervised_fits"]==1,"Missing local admission")
        native,volume=[],{}
        if tag!="replay":
            receipts=read(folder/"models_frozen.json")
            physical=[json.loads(line) for line in (folder/"physical.jsonl").read_text().splitlines()]
            require(physical==receipts["physical"] and len(physical)==3,"Native physical training differs")
            for i,r in enumerate(physical):
                worker=read(folder/f"member_{i}.fits.json")
                require(worker["candidate"]==r["candidate"] and worker["attempted_classifier_fits"]==
                        worker["completed_classifier_fits"]==1 and worker["status"]=="completed","Incomplete native fit")
            native=receipts["models"]
            for arm in ("small","pooled"):
                sub=study/"feedback_volume"/tag
                learned=read(sub/arm/"soft.json")
                require(sha(sub/arm/"soft.admission.json")==learned["admission_sha256"] and
                        read(sub/(arm+".json"))["policy"]==learned["policy"],"Volume gate not bound to its fit")
                volume[arm]=learned["policy"]["supervised_fits"]
        records.append({"scope":tag,"base_fits":len(native),"native":native,"gate_fits":recorded,"volume_gate_fits":volume})
    return {"paths":records,"native_total":sum(r["base_fits"] for r in records),
            "rule_total":sum(sum(r["gate_fits"].values())+sum(r["volume_gate_fits"].values()) for r in records),
            "model_api_calls":0,"llm_input_tokens":0,"llm_output_tokens":0,"new_course_rmb":0,
            "official_validation_used":False,"official_test_used":False}


def agreement_disputes(cols, labels):
    use = np.asarray(cols["old_unanimous"], dtype=bool) & np.asarray(cols["third_conflict"], dtype=bool)
    right = int((cols["third_alone"][use] == labels[use]).sum())
    return {"rows": int(use.sum()), "third_right": right, "original_pair_right": int(use.sum()) - right}


def publish(study, native_run):
    files=["protocol.json","historical_inputs.json","five_predictions_frozen.json","all_handoffs_frozen.json",
           "five_fold_results.json","local_five_fold_results.json","feedback_volume/protocol.json",
           "feedback_volume/all_sealed.json","feedback_volume/result.json"]
    for name in ("first","expanded"):
        files.extend(name+"/"+s for s in ("policy.json","predictions.csv","predictions_frozen.json","result.json"))
    files.extend("local_replay/"+s for s in ("decision.json","predictions_0.csv","predictions_frozen.json","result.json"))
    for i in range(5):
        f=f"fold_{i}/"
        files.extend(f+s for s in ("models_frozen.json","scores_frozen.json","probabilities.npz",
            "routing/policy.json","routing/predictions_0.csv","routing/predictions_frozen.json",
            "local/decision.json","local/predictions_0.csv","local/predictions_frozen.json"))
        files.extend("feedback_volume/"+f+s for s in ("small.json","pooled.json","predictions_0.csv","sealed.json"))
    for name in files:
        dest=PUBLIC/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        require(not dest.exists() or sha(dest)==sha(study/name),"Published evidence differs")
        shutil.copyfile(study/name,dest)
    save(PUBLIC/"accounting.json",accounting(study))
    decision = read(native_run/"conditional_handoff/decision.json")
    admission = read(native_run/"conditional_handoff/admission.json")
    digest_pair = [sha(study/f"delivery_{i}/predictions.csv") for i in range(2)]
    source = read(PUBLIC_BASE/"titles/completion.json")
    original_digest = sha(native_run/"audit/predictions.csv")
    with (study/"delivery_0/predictions.csv").open() as stream:
        ids = [int(row["row_id"]) for row in csv.DictReader(stream)]
    require(ids == list(range(source["independent_training_audit"]["rows"])) and
            digest_pair == [original_digest]*2 and original_digest == source["independent_training_audit"]["prediction_sha256"],
            "Real optional prediction differs from frozen native training reserve")
    save(PUBLIC/"optional_delivery.json", {"decision": decision, "admission": admission, "prediction_sha256": digest_pair,
         "rows": len(ids), "runtime_sha256": sha(ROOT/"tools/unified_handoff.py"),
         "cli_sha256": sha(ROOT/"scripts/unified_handoff.py")})
    contents=sorted(p for p in PUBLIC.rglob("*") if p.is_file() and p.name!="SHA256SUMS")
    (PUBLIC/"SHA256SUMS").write_text("".join(sha(p)+"  "+p.relative_to(PUBLIC).as_posix()+"\n" for p in contents))


def verify(archive):
    sha_list={s[66:]:s[:64] for s in (PUBLIC/"SHA256SUMS").read_text().splitlines()}
    actual={p.relative_to(PUBLIC).as_posix():sha(p) for p in PUBLIC.rglob("*") if p.is_file() and p.name!="SHA256SUMS"}
    require(sha_list==actual,"Public receipts do not match the checksum manifest")
    pr=read(PUBLIC/"protocol.json")
    require(pr["plan"]==sha(PLAN) and pr["gate"]==sha(GATE) and pr["prepare"]==sha(ROOT/"scripts/prepare_conditional_handoff.py")
            and pr["common_entry"]==sha(ROOT/"unified_agent.py"),"Common release or policy changed")
    volume_protocol=read(PUBLIC/"feedback_volume/protocol.json")
    require(volume_protocol["plan"]==sha(VOLUME_PLAN) and volume_protocol["runner"]==sha(ROOT/"scripts/study_handoff_feedback_volume.py"),
            "Feedback-volume protocol changed")
    archive_identity=read(EVIDENCE/"protocol.json")["archives"]["titles"]
    require(sha(archive)==archive_identity,"Wrong released training archive")
    rows=title_rows(archive,"train",labels=True)
    pairs=[r[:2] for r in rows]
    labels=np.array([r[2] for r in rows])
    group=sample_groups(SimpleNamespace(kind="pair_text",X_train=pairs),"shared_text")
    reserve=np.asarray(read(EVIDENCE/"title_reserve_ids.json"))
    old_disputes = agreement_disputes(load_export(PUBLIC/"expanded/predictions.csv"), labels[reserve])
    for name,file,suffix in (("first","predictions.csv","predictions"),("expanded","predictions.csv","predictions"),
                             ("local_replay","predictions_0.csv","predictions")):
        path=PUBLIC/name
        cols=load_export(path/file)
        seal=read(path/"predictions_frozen.json")[suffix]
        require((seal==sha(path/file) or seal==[sha(path/file)]*2) and
                cols["row_id"].tolist()==list(range(len(reserve))) and
                score_panel(cols,labels[reserve],group[reserve])==read(path/"result.json"),"Historical training replay differs")
    studies=[read(PUBLIC/"five_fold_results.json"),read(PUBLIC/"local_five_fold_results.json"),read(PUBLIC/"feedback_volume/result.json")]
    per_study=[{} for r in studies]
    all_ids=[]
    sets=list(partition(group,reserve))
    seals=read(PUBLIC/"all_handoffs_frozen.json")
    for i,roles in enumerate(sets):
        folder=PUBLIC/f"fold_{i}"
        volume=PUBLIC/"feedback_volume"/f"fold_{i}"
        expected=pr["splits"][i]["roles"]
        require(all(fingerprint(ix.tolist())==expected[k]["ids"] for k,ix in roles.items()),"Training grouping changed")
        ix=roles["outer"]
        all_ids.extend(ix)
        with np.load(folder/"probabilities.npz",allow_pickle=False) as packed:
            data={k:inputs(packed[k],[pairs[j] for j in roles[k]]) for k in ("learn","confirm","outer")}
        policy=read(folder/"routing/policy.json")
        rules=policy["policies"]
        require(policy["mechanism"]==sha(GATE) and policy["runner"]==sha(STUDY) and policy["plan"]==sha(PLAN)
                and select(rules,data["learn"],labels[roles["learn"]],data["confirm"],labels[roles["confirm"]])==
                policy["decision"],"Selection no longer follows its measured training feedback")
        checked=predictions(data["outer"],rules,policy["decision"])
        local=read(folder/"local/decision.json")
        require(local["plan"]==sha(LOCAL_PLAN) and local["study"]==sha(LOCAL_STUDY) and local["model_source"]==sha(POLICY_SOURCE)
                and current_rule(policy["decision"],local["eligible"],local["confirmation"])==local["chosen"],"Local admission changed")
        require(seals[i]=={"original":sha(folder/"routing/predictions_frozen.json"),"local":sha(folder/"local/predictions_frozen.json")},
                "Global prediction seal differs")
        cols0=load_export(folder/"routing/predictions_0.csv")
        require(all(np.array_equal(col,cols0[name]) for name,col in checked.items()),"Fresh-process predictions differ from saved rules")
        sources=(folder/"routing",folder/"local",volume)
        for j,(source,study) in enumerate(zip(sources,studies,strict=True)):
            pred=source/"predictions_0.csv"
            seal=read(source/("sealed.json" if j==2 else "predictions_frozen.json"))
            require(seal.get("predictions",seal.get("prediction_replicas"))==[sha(pred)]*2,"Repeated prediction digest differs")
            cols=load_export(pred)
            require(cols["row_id"].tolist()==ix.tolist(),"Outer rows changed")
            if j==1:
                desired=cols["local"] if local["chosen"]=="local" else cols["previous_agent"]
                require(np.array_equal(cols["agent"],desired),"Wrong extended decision")
            if j==2:
                for arm in ("small","pooled"):
                    dec=read(source/(arm+".json"))["selection"]
                    desired=cols[arm+"_shadow"] if dec["chosen"]=="soft" else cols["baseline"]
                    require(np.array_equal(cols[arm],desired),"Wrong feedback-volume decision")
            measured=score_panel(cols,labels[ix],group[ix])
            require(all(study["folds"][i][k]==v for k,v in measured.items()),"Per-fold score or interval differs")
            for k,col in cols.items():
                per_study[j].setdefault(k,[]).append(col)
    all_ids=np.asarray(all_ids)
    require(sorted(np.r_[all_ids,reserve].tolist())==list(range(len(rows))),"Fresh evaluation groups overlap or are missing")
    for j,(study,saved) in enumerate(zip(studies,per_study,strict=True)):
        cols={k:np.concatenate(v) for k,v in saved.items()}
        if j == 0:
            new_disputes = agreement_disputes(cols, labels[all_ids])
        measured=score_panel(cols,labels[all_ids],group[all_ids])
        positive=sum(r["methods"]["agent"]["net"]>0 for r in study["folds"])
        stats=measured["methods"]["agent"]
        gate=positive>=4 and stats["net"]>=5 and stats["paired_95_fraction"][0]>0
        if j==2:
            effect=comparison(cols["small"],cols["pooled"],labels[all_ids],group[all_ids])
            require(effect==study["extra_feedback_effect"],"Effect of extra feedback differs")
            gate=gate and effect["net"]>0
        require(measured==study["pooled"] and positive==study["positive_folds"] and bool(gate)==study["passes"],
                "Pooled score, paired uncertainty or promotion changed")
    costs=read(PUBLIC/"accounting.json")
    require(costs["native_total"]==sum(len(p["native"]) for p in costs["paths"])==15 and
            costs["rule_total"]==sum(sum(p["gate_fits"].values())+sum(p["volume_gate_fits"].values()) for p in costs["paths"])
            and costs["model_api_calls"]==0,"Usage differs")
    optional = read(PUBLIC/"optional_delivery.json")
    decision, prior = optional["decision"], read(PUBLIC_BASE/"titles/completion.json")
    original = prior["independent_training_audit"]
    replay = read(PUBLIC/"expanded/policy.json")
    require(decision["chosen"] == "baseline" and decision["model_sha256"] == prior["model"]
            and decision["members"] == prior["members"] and decision["challenger"] == pr["members"][2]
            and decision["policy"] == replay["policies"]["soft"]
            and decision["decision"]["evidence"]["soft"] == replay["decision"]["evidence"]["soft"]
            and decision["extra_fits"] == 1 and decision["prior_native_fits"] == 10
            and optional["admission"] == {"maximum_extra_fits": 1, "prior_native_fits": 10, "path_limit": 11}
            and optional["rows"] == original["rows"] and optional["prediction_sha256"] == [original["prediction_sha256"]]*2
            and optional["runtime_sha256"] == sha(ROOT/"tools/unified_handoff.py")
            and optional["cli_sha256"] == sha(ROOT/"scripts/unified_handoff.py"), "Optional native handoff differs")
    print("When both original members agreed and the third disagreed:", old_disputes, "on the historical reserve;",
          new_disputes, "on the five new grouped folds.")
    print(f"Verified the original training reserve, {len(all_ids)} disjoint outer rows, three paired studies, "
          f"{costs['native_total']} native and {costs['rule_total']} study routing fits, and one deployment fit; "
          "opened train.csv only.")


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--publish",type=Path)
    p.add_argument("--native-run",type=Path,help="Completed raw training run used for the optional delivery check")
    p.add_argument("--archive",type=Path,required=True)
    a=p.parse_args()
    if a.publish:
        if not a.native_run:
            p.error("--publish requires --native-run")
        publish(a.publish, a.native_run)
    verify(a.archive)
