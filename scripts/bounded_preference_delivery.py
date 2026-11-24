"""Package an existing verified three-member binary vote for label-only bounded inference."""
import argparse
import os
from pathlib import Path
import shutil
import tempfile

import joblib
import numpy as np

from scripts.unified_released_study import read, require, save, sha
from superagent.ab_complement import PreferenceProbabilityVote
from superagent.bounded_vote import BoundedPreferenceVote


def package(source_run, destination_run):
    src = source_run / "agent"
    selection = read(src / "final_selection.json")
    spec = read(src / "task_spec.json")
    require(selection["model_file"] == "selected_model.joblib" and selection["model_format"] == "candidate_bundle_v1"
            and selection["model_sha256"] == sha(src / "selected_model.joblib")
            and selection["task_spec_sha256"] == sha(src / "task_spec.json"), "Selected model or contract changed")
    require(spec.get("kind") == "pair_text" and spec.get("input_schema", {}).get("pair_semantics") == "preference"
            and len(selection.get("ensemble_components", [])) == 3, "This delivery does not declare a three-member preference vote")
    bundle = joblib.load(src / "selected_model.joblib")
    original = bundle["model"]
    require(type(original) is PreferenceProbabilityVote and original.kind == "pair_text"
            and np.array_equal(original.classes_, [0, 1]) and len(original.models) == 3, "Unsupported underlying vote")
    if destination_run.exists():
        raise FileExistsError(destination_run)
    destination_run.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination_run.parent) as tmp:
        partial = Path(tmp) / "ready"
        agent = partial / "agent"
        agent.mkdir(parents=True)
        shutil.copyfile(src / "task_spec.json", agent / "task_spec.json")
        with (agent / "selected_model.joblib").open("xb") as output:
            joblib.dump({**bundle, "model": BoundedPreferenceVote(original.models)}, output, compress=1)
        save(agent / "final_selection.json", {**selection, "model_sha256": sha(agent / "selected_model.joblib"),
                                               "prediction_optimization": "bounded_three_vote_v1",
                                               "original_model_sha256": selection["model_sha256"]})
        os.rename(partial, destination_run)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("completed_run", type=Path)
    p.add_argument("new_prediction_run", type=Path)
    args = p.parse_args()
    package(args.completed_run, args.new_prediction_run)
