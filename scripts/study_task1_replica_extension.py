"""Measure the previously fixed inference rule on both remaining development folds."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.study_task1_outer import compare, prediction_csv
from scripts.unified_released_study import ROOT, read, require, save, sha

PLAN = ROOT / "research/task1_feedback_extension_1127/remaining_replica_diagnostic.json"


def grade(out):
    folds = [(i, out / ("fusion_match" if i < 3 else "near_full") / str(i)) for i in range(5)]
    alternatives = ("weighted", "replicas")
    sealed = {}
    for i, root in folds:
        record = read(root / "reuse/sealed.json")
        require(record["old_prediction_rebuilt"] and record["training_fits"] == 0
                and record["decision"] == sha(root / "reuse/decision.json"), "Unsealed replica inference")
        for name in alternatives:
            require(sha(root / "reuse" / (name+".npy")) == record["predictions"][name], "Changed diagnostics")
        sealed[str(i)] = sha(root / "reuse/sealed.json")
    save(out / "replica_five_lock.json", dict(protocol=sha(PLAN), seals=sealed))
    rows = []
    records = {a: [] for a in ("ordinary", *alternatives)}
    groups = []
    for i, root in folds:
        truth = root / "outer_answers.joblib"
        require(sha(truth) == read(root / "inputs.json")["outer_answers"], "Unbound answers")
        key = joblib.load(truth)
        y = key["y"].astype(str)
        reference = prediction_csv(root / "regular/solve.csv", len(y)) == y
        results = {"ordinary": reference}
        for name in alternatives:
            results[name] = np.load(root / "reuse" / (name+".npy"), allow_pickle=False).astype(str) == y
        for name, correct in results.items():
            records[name].extend(correct)
        groups.extend(key["groups"])
        rows.append(dict(fold=i, rows=len(y), scores={k: int(v.sum()) for k,v in results.items()}))
    result = dict(lock=sha(out / "replica_five_lock.json"), folds=rows, rows=len(groups),
                  totals={k:int(np.sum(v)) for k,v in records.items()},
                  differences={a: compare(np.asarray(records[a]), np.asarray(records["ordinary"]), groups)
                               for a in alternatives}, fits=0, api_requests=0)
    save(out / "replica_five_result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=lambda p: Path(p).resolve())
    grade(parser.parse_args().out)
