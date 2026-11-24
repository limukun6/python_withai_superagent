"""Learn, freeze and audit routing decisions using four disjoint training roles."""
import argparse
import csv
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np

from scripts.prepare_conditional_handoff import GATE, PLAN, PUBLIC_BASE, ROLES, members
from scripts.unified_confidence_study import paired_interval
from scripts.unified_released_study import EVIDENCE, ROOT, read, require, save, sha
from scripts.unified_third_member import same_split
from superagent.data import load_prediction_features, load_training_task
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups
from tools.preference_handoff import POLICIES, changes, fit_policy, inputs, predictions, select

STUDY = Path(__file__).resolve()


def historical_panels(out, run):
    names, _ = members(run)
    agent = run / "agent"
    task = load_training_task(run / "training.zip")
    group = sample_groups(task, "shared_text")
    records = read(agent / "prediction_evidence.json")
    panels, targets, meta = {}, {}, {}
    prior = set()
    for role, split in (("learn", "primary"), ("confirm", "confirmation_1")):
        ys, _, observed = same_split(agent, split, records)
        ids = np.asarray(read(agent / (split + "_split.json"))["search_indices"])
        require(np.array_equal(task.y_train[ids], ys), "Native training feedback misaligned")
        valid = np.array([v not in prior for v in group[ids]])
        prior.update(group[ids])
        kept = ids[valid]
        x = np.column_stack([observed[k][0] for k in names])[valid]
        panels[role] = inputs(x, [task.X_train[i] for i in kept])
        targets[role] = ys[valid]
        meta[role] = {"rows": len(kept), "excluded": int((~valid).sum()), "groups": len(set(group[kept])),
                      "ids": fingerprint(kept.tolist()), "native_split": split}
    source=read(out / "historical_inputs.json")
    file=run/"audit_features.zip"
    require(source["features"]==sha(file) and source["probabilities"]==sha(out/"historical_probabilities.npy"),
            "Historical scores or features differ")
    pairs=load_prediction_features(file,"pair_text",read(agent/"task_spec.json"))
    panels["outer"]=inputs(np.load(out/"historical_probabilities.npy",allow_pickle=False), pairs)
    with (run/"audit/predictions.csv").open() as f:
        raw = list(csv.DictReader(f))
    require(np.array_equal([int(r["prediction"]) for r in raw], panels["outer"]["base"])
            and [int(r["row_id"]) for r in raw] == list(range(len(pairs))), "Original delivered prediction changed")
    return panels, targets, meta


def learned(path, name, train, labels):
    path.mkdir(parents=True,exist_ok=True)
    rec=path/(name+".json")
    identity=fingerprint({"probabilities":train["p"].tolist(), "features":train["features"].tolist(),
                          "labels":labels.tolist(),"policy":name,"gate_code":sha(GATE)})
    if rec.exists():
        record=read(rec)
        require(record["input"]==identity,"Cached supervised policy was trained on different feedback")
        return record["policy"]
    admission=path/(name+".admission.json")
    with admission.open("x") as f:
        f.write('{"maximum_supervised_fits":1}\n')
    policy=fit_policy(name,train,labels)
    save(rec,{"input":identity,"policy":policy,"admission_sha256":sha(admission)})
    return policy


def freeze_policy(destination, fitting, data, labels, strategies, description):
    destination.mkdir(parents=True,exist_ok=True)
    policies={name:learned(fitting,name,data["learn"],labels["learn"]) for name in strategies}
    decision=select(policies,data["learn"],labels["learn"],data["confirm"],labels["confirm"])
    bundle={"policies":policies,"decision":decision,"member_order":"two frozen controls, then a third expert",
            "mechanism":sha(GATE),"runner":sha(STUDY),"plan":sha(PLAN),"feedback_panels":description}
    save(destination/"policy.json",bundle)


def write_predictions(path, bundle, data, ids):
    require(bundle["mechanism"]==sha(GATE),"Serialized conditional policy differs")
    cols=predictions(data,bundle["policies"],bundle["decision"])
    fields=["row_id","old_unanimous","third_conflict",*cols]
    with path.open("x",newline="",encoding="utf-8") as stream:
        writer=csv.writer(stream)
        writer.writerow(fields)
        for i,row_id in enumerate(ids):
            writer.writerow([int(row_id),int(data["unanimous"][i]),int(data["conflict"][i]),*[int(col[i]) for col in cols.values()]])


def load_export(path):
    with path.open(encoding="utf-8") as stream:
        rows=list(csv.DictReader(stream))
    return {k:np.asarray([int(row[k]) for row in rows]) for k in rows[0]}


def comparison(base,predicted,y,groups):
    stats=changes(base,predicted,y)
    _,codes=np.unique(groups,return_inverse=True)
    sizes=np.bincount(codes)
    delta=np.bincount(codes,weights=(predicted==y).astype(int)-(base==y).astype(int)).astype(int)
    return {**stats,"paired_95_fraction":paired_interval(sizes,delta)}


def score_panel(cols, truth, groups):
    y=np.asarray(truth)
    base=cols["baseline"]
    shared=(base!=y) & cols["old_unanimous"].astype(bool)
    measures={}
    for name, col in cols.items():
        if name in ("row_id","old_unanimous","third_conflict"):
            continue
        metrics=comparison(base,col,y,groups) if name!="baseline" else {**changes(base,base,y),"paired_95_fraction":[0.,0.]}
        metrics["common_errors_recovered"]=int((shared & (col==y)).sum())
        measures[name]=metrics
    require(np.all(cols["agent"][cols["third_conflict"]==0]==base[cols["third_conflict"]==0]),"Changed a non-conflict")
    return {"rows":len(y),"groups":len(np.unique(groups)), "two_members_both_wrong":int(shared.sum()),
            "conflicts_with_third":int(cols["third_conflict"].sum()),"methods":measures,
            "agent_versus_equal_three":comparison(cols["equal_three"],cols["agent"],y,groups)}


def replay(out, run, train_study, name):
    if name=="expanded":
        require((out/"first/result.json").is_file(),"Finish the hard-gate study first")
    panels, targets, meta=historical_panels(out,run)
    destination=out/name
    freeze_policy(destination,out/"replay_fits",panels,targets,POLICIES[:1] if name=="first" else POLICIES,meta)
    policy=read(destination/"policy.json")
    seal_path=destination/"predictions_frozen.json"
    if not seal_path.exists():
        require(not (destination/"predictions.csv").exists(),"Previous incomplete replay export")
        write_predictions(destination/"predictions.csv",policy,panels["outer"],range(len(panels["outer"]["base"])))
        save(seal_path,{"predictions":sha(destination/"predictions.csv"),"policy":sha(destination/"policy.json"),
                        "features":read(out/"historical_inputs.json")["features"]})
    # Only the completed prediction stage is allowed to join the historical training targets.
    require(sha(destination/"predictions.csv")==read(seal_path)["predictions"] and
            sha(train_study/"development_truth.json")==read(train_study/"protocol.json")["development_truth"] and
            read(run/"training_protocol.json")["audit_row_ids_sha256"]==fingerprint(read(EVIDENCE/"title_reserve_ids.json")),
            "Wrong historical training reserve")
    truth=read(train_study/"development_truth.json")
    result=score_panel(load_export(destination/"predictions.csv"),truth["labels"],truth["groups"])
    save(destination/"result.json",result)
    print(name,policy["decision"]["chosen"],{k:v["net"] for k,v in result["methods"].items()},flush=True)


def fresh(out, n):
    folder=out/f"fold_{n}"
    source=read(out/"protocol.json")
    expected=source["splits"][n]
    require(sha(folder/"feedback.npz")==expected["feedback"] and sha(folder/"features.joblib")==expected["features"]
            and read(folder/"scores_frozen.json")["scores"]==sha(folder/"probabilities.npz"),"Fresh evidence changed")
    obs=joblib.load(folder/"features.joblib")
    with np.load(folder/"probabilities.npz",allow_pickle=False) as p, np.load(folder/"feedback.npz",allow_pickle=False) as ys:
        data={r:inputs(p[r],obs[r]["pairs"]) for r in ("learn","confirm")}
        labels={r:ys[r] for r in data}
    target=folder/"routing"
    freeze_policy(target,folder/"gate_fits",data,labels,POLICIES,{k:expected["roles"][k] for k in ("learn","confirm")})
    files=[]
    for copy in range(2):
        output=target/f"predictions_{copy}.csv"
        if not output.exists():
            subprocess.run([sys.executable,"-m","scripts.run_conditional_handoff","predict","--out",str(out),
                "--fold",str(n),"--output",str(output)],cwd=ROOT,stdin=subprocess.DEVNULL,check=True,timeout=60)
        files.append(sha(output))
    require(files[0]==files[1],"Serialized router changed predictions on replay")
    save(target/"predictions_frozen.json",{"prediction_replicas":files,"policy":sha(target/"policy.json"),
                                           "scores":sha(folder/"probabilities.npz"),"features":sha(folder/"features.joblib")})


def predict_only(out,n,output):
    folder=out/f"fold_{n}"
    expected=read(out/"protocol.json")["splits"][n]
    require(sha(folder/"features.joblib")==expected["features"] and
            read(folder/"scores_frozen.json")["scores"]==sha(folder/"probabilities.npz"),"Prediction input differs")
    obs=joblib.load(folder/"features.joblib")["outer"]
    with np.load(folder/"probabilities.npz",allow_pickle=False) as p:
        data=inputs(p["outer"],obs["pairs"])
    write_predictions(output,read(folder/"routing/policy.json"),data,obs["ids"])


def fresh_score(out):
    protocol=read(out/"protocol.json")
    seals=[read(out/f"fold_{n}/routing/predictions_frozen.json") for n in range(5)]
    save(out/"five_predictions_frozen.json",seals)
    combined, all_y, all_groups={}, [], []
    details=[]
    for n,seal in enumerate(seals):
        folder=out/f"fold_{n}"
        predictions_file=folder/"routing/predictions_0.csv"
        require(seal["prediction_replicas"]==[sha(predictions_file),sha(folder/"routing/predictions_1.csv")]
                and seal["policy"]==sha(folder/"routing/policy.json"),"Unsealed outer decisions")
        target_file=out/"scoring_only"/f"fold_{n}.json"
        require(sha(target_file)==protocol["splits"][n]["targets"],"Sealed targets differ")
        target=read(target_file)
        cols=load_export(predictions_file)
        require(cols["row_id"].tolist()==target["ids"],"Wrong outer row order")
        require(not set(target["groups"]).intersection(all_groups),"A headline component appears in two outer folds")
        panel=score_panel(cols,target["labels"],target["groups"])
        details.append({"fold":n,"chosen":read(folder/"routing/policy.json")["decision"]["chosen"],**panel})
        all_y.extend(target["labels"])
        all_groups.extend(target["groups"])
        for k,v in cols.items():
            combined.setdefault(k,[]).append(v)
    merged={k:np.concatenate(v) for k,v in combined.items()}
    require(len(np.unique(merged["row_id"]))==len(all_y)==protocol["outer_rows"],"Duplicate or missing outer row")
    total=score_panel(merged,all_y,all_groups)
    positive=sum(r["methods"]["agent"]["net"]>0 for r in details)
    actual=total["methods"]["agent"]
    result={"folds":details,"pooled":total,"positive_folds":positive,"passes":bool(positive>=4 and actual["net"]>=5
            and actual["paired_95_fraction"][0]>0),"course_validation_used":False,"course_test_used":False,"model_api_calls":0}
    save(out/"five_fold_results.json",result)
    print("folds",[(r["chosen"],r["methods"]["agent"]["net"]) for r in details],"pooled",actual,"gate",result["passes"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("first","expanded","freeze","predict","score"))
    for k in ("out","run","train-study","output"):
        p.add_argument("--"+k,required=k=="out",type=lambda s:Path(s).resolve())
    p.add_argument("--fold",type=int)
    a=p.parse_args()
    if a.stage in ("first","expanded"):
        replay(a.out,a.run,a.train_study,a.stage)
    elif a.stage=="freeze":
        require((a.out/"expanded/result.json").is_file(),"Freeze the expanded policy before external scoring")
        for fold in ([a.fold] if a.fold is not None else range(5)):
            fresh(a.out,fold)
    elif a.stage=="predict":
        predict_only(a.out,a.fold,a.output)
    else:
        fresh_score(a.out)
