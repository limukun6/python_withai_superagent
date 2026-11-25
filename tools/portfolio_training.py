"""Budgeted fitting and native export shared by the anonymous entry and the grouped comparison."""
import os
from pathlib import Path

import joblib

from scripts.unified_released_study import read, require, sha
from superagent.ab_methods import TEXT, TEXT_FAMILIARITY, TRANSFER_METHODS
from superagent.ab_sentence import fit_sentence
from superagent.fit_guard import admitted_fits
from tools.encoder_assets import MODELS
from tools.portfolio_encoders import fit_head
from tools.preference_portfolio import delivery
from unified_agent import save


REFERENCE = (TEXT_FAMILIARITY, TRANSFER_METHODS["deep-e5"][1], TEXT)


def configure(weights, cache=None):
    weights = Path(weights).resolve()
    os.environ.update(SUPERAGENT_PORTFOLIO_WEIGHTS=str(weights), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      TOKENIZERS_PARALLELISM="false", SUPERAGENT_OFFLINE="1")
    for key, sub in (("MINILM", "all-MiniLM-L6-v2"), ("DEEP_MINILM", "all-MiniLM-L12-v2"),
                     ("E5_SMALL", "e5-small-v2"), ("BGE_SMALL", "bge-small-en-v1.5")):
        os.environ["SUPERAGENT_" + key + "_WEIGHTS"] = str(weights / sub)
    if cache is None:
        os.environ.pop("SUPERAGENT_PORTFOLIO_CACHE", None)
    else:
        os.environ["SUPERAGENT_PORTFOLIO_CACHE"] = str(Path(cache).resolve())


def fit_one(folder, name, rows, labels, identity):
    if (folder/"trained.json").exists():
        record = read(folder/"trained.json")
        require(record["identity"] == identity and record["name"] == name and record["model"] == sha(folder/"model.joblib")
                and record["ledger"] == sha(folder/"fits.json"), "Cached learner does not match the supplied training rows")
        return record
    folder.mkdir(parents=True, exist_ok=False)
    save(folder/"admission.json", {"maximum_classifier_fits": 1, "training_identity": identity, "candidate": name})
    with admitted_fits({"maximum_classifier_fits": 1, "ledger_path": str(folder/"fits.json")}, name) as meter:
        if name in MODELS:
            model = fit_head(name, rows, labels)
        elif name in REFERENCE:
            model, _ = fit_sentence(rows, labels, rows[:1], "pair_text", familiarity=name == REFERENCE[0],
                                    alternative="e5" if name == REFERENCE[1] else None)
        else:
            raise ValueError(f"Unknown candidate {name}")
    require(meter["attempted_classifier_fits"] == meter["completed_classifier_fits"] == 1, "Classifier budget differs")
    with (folder/"model.joblib").open("xb") as output:
        joblib.dump(model, output, compress=1)
    record = {"name": name, "identity": identity, "model": sha(folder/"model.joblib"), "ledger": sha(folder/"fits.json"),
              "fits": meter["calls"], "model_api_calls": 0}
    save(folder/"trained.json", record)
    return record


def export(directory, models, choice, specification):
    directory.mkdir(parents=True, exist_ok=False)
    spec = dict(specification, task_kind="pair_text_binary", kind="pair_text", metric="accuracy", test_accessed=False,
                metric_spec={"name": "accuracy", "labels": [0, 1], "direction": "maximize"})
    spec.setdefault("input_columns", spec["input_schema"]["columns"])
    save(directory/"task_spec.json", spec)
    family = "feedback_selected_preference_portfolio"
    with (directory/"selected_model.joblib").open("xb") as stream:
        joblib.dump({"kind": "pair_text", "candidate": family, "model": delivery(models, choice)}, stream, compress=1)
    rule = choice["rules"][choice["chosen"]]
    used = [choice["members"][i] for i in rule["indices"]] if rule["kind"] == "subset" else choice["members"]
    save(directory/"final_selection.json", {"family": family, "selected_experiment": choice["chosen"],
         "model_format": "candidate_bundle_v1", "model_file": "selected_model.joblib",
         "model_sha256": sha(directory/"selected_model.joblib"), "task_spec_sha256": sha(directory/"task_spec.json"),
         "test_accessed": False, "ensemble_components": used})
