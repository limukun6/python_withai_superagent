"""Train the registered new heads on original grouped base rows and freeze their unlabeled predictions."""
import argparse
from pathlib import Path

import joblib
import numpy as np

from scripts.prepare_conditional_handoff import ROLES
from scripts.unified_released_study import ROOT, read, require, sha
from superagent.ab_complement import canonical_probabilities
from tools.encoder_assets import MODELS, verified
from tools.portfolio_training import configure, fit_one
from unified_agent import save


PRIOR = ROOT/"research/unified_handoff_1124"
PLAN = ROOT/"docs/batches/unified_encoder_fusion_plan_20261124_zh.md"


def fit_fold(prior, out, weights, cache, name, index):
    configure(weights, cache)
    verified(weights, name)
    spec = read(PRIOR/"protocol.json")
    require(sha(prior/"protocol.json") == sha(PRIOR/"protocol.json"), "Source fold protocol differs")
    original = prior/f"fold_{index}"
    expected = spec["splits"][index]
    payload, features = original/"base.joblib", original/"features.joblib"
    require(sha(payload) == expected["payload"] and sha(features) == expected["features"], "Base or feedback rows differ")
    bank = cache/(name + ".npz")
    require(bank.is_file(), "The frozen feature bank is not ready")
    dest = out/f"fold_{index}"/name
    frozen = dest/"scores.json"
    identity = {"base": expected["payload"], "features": expected["features"], "cache": sha(bank), "plan": sha(PLAN),
                "fitter": sha(Path(__file__)), "training": sha(ROOT/"tools/portfolio_training.py"),
                "features_code": sha(ROOT/"tools/portfolio_encoders.py")}
    if frozen.exists():
        result = read(frozen)
        require(result["identity"] == identity and result["scores"] == sha(dest/"probabilities.npz")
                and result["model"] == sha(dest/"model.joblib"), "Frozen new model is inconsistent")
        return
    base = joblib.load(payload)
    trained = fit_one(dest, name, base["X_train"], base["y_train"], identity)
    panels = joblib.load(features)
    model = joblib.load(dest/"model.joblib")
    probabilities = {r: canonical_probabilities(model, panels[r]["pairs"])[1][:, 1] for r in ROLES}
    tmp = dest/"probabilities.pending.npz"
    np.savez(tmp, **probabilities)
    tmp.replace(dest/"probabilities.npz")
    save(frozen, {"name": name, "identity": identity, "model": trained["model"], "ledger": trained["ledger"],
                  "scores": sha(dest/"probabilities.npz"), "rows": {r: len(probabilities[r]) for r in ROLES}})
    print(index, name, "one classifier fit; predictions frozen", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("prior", "out", "cache", "weights"):
        p.add_argument("--"+key, type=Path, required=True)
    p.add_argument("--model", choices=tuple(MODELS), required=True)
    p.add_argument("--folds", nargs="+", type=int, choices=range(5), default=list(range(5)))
    a = p.parse_args()
    for i in a.folds:
        fit_fold(a.prior, a.out, a.weights, a.cache, a.model, i)
