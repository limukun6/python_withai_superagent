"""Check unseen-input execution without the opportunity to read any cached title vector."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.unified_released_study import read, require, sha
from superagent.ab_complement import canonical_probabilities
from tools.encoder_assets import MODELS
from tools.portfolio_training import configure
from unified_agent import save


def check(study, prior, weight_path):
    configure(weight_path, cache=None)
    stage = study/"fold_0"
    inputs = joblib.load(prior/"fold_0/features.joblib")["outer"]["pairs"]
    indexes = np.linspace(0, len(inputs)-1, 32, dtype=int)
    rows = [inputs[i] for i in indexes]
    reports = {}
    for name in MODELS:
        local = stage/name
        with np.load(local/"probabilities.npz", allow_pickle=False) as p:
            cached = p["outer"][indexes]
        path = local/"model.joblib"
        require(sha(path) == read(local/"scores.json")["model"], "Wrong independently fitted head")
        model = joblib.load(path)
        actual = canonical_probabilities(model, rows)[1][:, 1]
        reverse = canonical_probabilities(model, [(b,a) for a,b in rows])[1][:, 1]
        err = float(np.max(np.abs(actual-cached)))
        require(err < 2e-5 and np.array_equal(actual>.5,cached>.5) and np.allclose(actual+reverse,1,atol=1e-9),
                "Online encoder / frozen cache or pair symmetry differs")
        reports[name] = {"rows":len(rows),"maximum_probability_difference":err,"same_classes":True,"swapped":True}
    value = {"models":reports,"cache_used_at_reload":False,"new_classifier_fits":0,"model_api_calls":0}
    if (study/"feature_reload.json").exists():
        require(read(study/"feature_reload.json")==value,"Repeated audit differs")
    else:
        save(study/"feature_reload.json",value)
    print(value)


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("study","prior","weights"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    check(a.study,a.prior,a.weights)
