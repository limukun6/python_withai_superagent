"""Verify actual actions and test sensitivity to different row errors at identical accuracy."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np

from superagent.confirmation import shortlist
from superagent.method_choice import select_action
from tools.frontier_study import expect
from tools.pretrained_study import read, write


def action(record, evidence):
    return select_action(evidence,record["eligible"],record["mode"],record["previous_methods"])[0]


def inspect_run(folder):
    history=read(folder/"trials.json")
    initial=shortlist(history[:5],5)
    truth=joblib.load(folder/"inner_data.joblib")["y_search"]
    vectors={r["candidate"]:joblib.load(folder/f"trial_{r['trial']}.joblib")["search_predictions"] for r in history}
    anchor_correct=vectors[initial[0]["candidate"]]==truth
    common=np.logical_and.reduce([vectors[r["candidate"]]!=truth for r in initial])
    checks=[]
    for i,file in enumerate(sorted(folder.glob("method_decision_*.json"))):
        rec=read(file)
        if "evidence" not in rec:continue
        ev=rec["evidence"]
        expect(ev["shared_errors"]==int(common.sum()) and ev["search_rows"]==len(truth),"Inconsistent shared errors")
        actual=action(rec,ev)
        expect(actual==rec.get("candidate") and (rec["decision"]=="try")==bool(actual),"Frozen decision cannot be replayed")
        if actual:
            expect(history[5+i]["candidate"]==actual,"Planned and executed candidate differ")
        if i!=1 or ev["primary_metric"]!="accuracy":
            checks.append(dict(actual=actual));continue
        measured=vectors[history[5]["candidate"]]==truth
        gain=int(np.sum(measured & ~anchor_correct))
        loss=int(np.sum(~measured & anchor_correct))
        expect((gain,loss)==(ev["corrected"],ev["lost"]) and gain+loss==ev["discordant"],"Inconsistent paired errors")
        delta=gain-loss
        expect(abs(delta/len(truth)-ev["previous_delta"])<1e-10,"Marginal accuracy changed")
        low=max(0,delta)
        high=min(int((~anchor_correct).sum()),int(anchor_correct.sum())+delta)
        def alternative(corrected):
            lost=corrected-delta
            return action(rec,ev|dict(corrected=corrected,lost=lost,discordant=corrected+lost))
        checks.append(dict(actual=actual,lower=alternative(low),upper=alternative(high)))
    return checks


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    seen=set()
    stats=defaultdict(Counter)
    traces={}
    for path in sorted(args.root.glob("*/results.json")):
        protocol=read(path.parent/"protocol.json")
        if "locations" not in protocol:continue
        result=read(path)
        for row in protocol["locations"]:
            for arm,directory in row.items():
                source=Path(directory)
                if arm=="classic" or source in seen:continue
                seen.add(source)
                stats[arm]["runs"]+=1
                for decision in inspect_run(source):
                    stats[arm]["verified_decisions"]+=1
                    if "lower" in decision:
                        stats[arm]["accuracy_decisions_tested"]+=1
                        variants={decision[key] for key in ("actual","lower","upper")}
                        stats[arm]["same_score_different_action"]+=int(len(variants)>1)
                        stats[arm]["actual_differs_from_min_disagreement"]+=int(decision["actual"]!=decision["lower"])
                        stats[arm]["actual_differs_from_max_disagreement"]+=int(decision["actual"]!=decision["upper"])
        for adaptive,fixed in (("paired","context-control"),("anchored","anchored-control"),("text-encoder","text-encoder-control"),("encoder-depth","encoder-depth-control")):
            if not {adaptive,fixed}<=result["correct"].keys():continue
            matched=[]
            for fold in result["folds"]:
                a,b=fold["arms"][adaptive],fold["arms"][fixed]
                matched.append(dict(fold=fold["fold"],identical_slots=a["extras"]==b["extras"],same_final_candidate=a["winner"]==b["winner"],
                                    gain=a["correct"]-b["correct"],rows=fold["rows"]))
            traces[f"{path.parent.name}/{adaptive}"]=matched
    report=dict(modes=dict(sorted(stats.items())),unique_runs=len(seen),matched_controls=traces,
                interpretation="Sensitivity holds the class labels, five initial models, eligible candidates, previous method and measured accuracy gain fixed. The error overlap varies over the feasible minimum and maximum for that same accuracy. Only actions are replayed. All accuracy claims come from complete measured runs.")
    write(args.output,report)
    print(len(seen),"unique runs;",sum(c["verified_decisions"] for c in stats.values()),"verified decisions")


if __name__=="__main__":main()
