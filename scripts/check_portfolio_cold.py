"""Seal anonymous native exports, then audit the held portion of the course training split."""
import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.run_conditional_handoff import comparison, load_export
from scripts.unified_released_study import EVIDENCE, read, require, sha, title_rows
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups
from tools.preference_portfolio import weights
from unified_agent import save


def audit_cold(cold, archive):
    pr, done, choice = (read(cold/f) for f in ("protocol.json", "completion.json", "decision.json"))
    deployed = [sha(cold/k/"predictions.csv") for k in ("cached", "uncached")]
    base_file, actual = cold/"reference_export/predictions.csv", cold/"cached/predictions.csv"
    require(deployed[0] == deployed[1] and done["model_sha256"] == sha(cold/"agent/selected_model.joblib") and
            done["classifier_fits"] == len(done["members"]) == len(choice["members"]), "Anonymous delivered model changed")
    for trained in done["members"]:
        ledger = cold/"members"/trained["name"]/"fits.json"
        require(sha(ledger) == trained["ledger"] and read(ledger)["attempted_classifier_fits"] ==
                read(ledger)["completed_classifier_fits"] == 1, "Real physical classifier budget differs")
    seal = {"deployed_predictions": deployed, "reference": sha(base_file), "decision": sha(cold/"decision.json"),
            "deployed_model": done["model_sha256"], "audit_features": sha(cold/"audit_features.zip")}
    if (cold/"audit_seal.json").exists():
        require(read(cold/"audit_seal.json") == seal, "Native delivery changed after scoring")
    else:
        save(cold/"audit_seal.json", seal)
    prior = read(EVIDENCE/"protocol.json")
    require(sha(archive) == prior["archives"]["titles"], "Wrong training archive")
    reserve = read(EVIDENCE/"title_reserve_ids.json")
    require(fingerprint(reserve) == pr["roles"]["audit"]["ids"], "Different independent audit rows")
    rows = title_rows(archive, "train", labels=True)
    pairs = [r[:2] for r in rows]
    groups = sample_groups(SimpleNamespace(kind="pair_text", X_train=pairs), "shared_text")
    labels = np.array([r[2] for r in rows])[reserve]
    control, candidate = load_export(base_file), load_export(actual)
    require(control["row_id"].tolist() == candidate["row_id"].tolist() == list(range(len(reserve))),
            "Native audit row order differs")
    result = comparison(control["prediction"], candidate["prediction"], labels, groups[reserve])
    rule = choice["rules"][choice["chosen"]]
    w = None
    if rule["kind"] == "constant":
        w = weights(rule, {"scores": np.full((1,len(choice["members"])),.5)})[0].tolist()
    result.update(chosen=choice["chosen"], members=choice["members"], fixed_weights=w,
                  seal=sha(cold/"audit_seal.json"), test_or_course_validation_used=False)
    if (cold/"audit_result.json").exists():
        require(read(cold/"audit_result.json") == result, "Anonymous reserve score differs")
    else:
        save(cold/"audit_result.json", result)
    print(choice["chosen"], result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cold",type=Path,required=True)
    p.add_argument("--archive",type=Path,required=True)
    a=p.parse_args()
    audit_cold(a.cold,a.archive)
