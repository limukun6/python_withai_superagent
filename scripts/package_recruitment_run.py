"""Export the verified course run with an explicitly unknown designated score."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT, predictions, read,require,save,sha
from superagent.external_assets import activate_resource_directory
from tools.package_verified_delivery import package


def run(root, features):
    arm=root/"recruitment-raw/recruitment_raw"
    seal,plan=read(arm/"sealed.json"),read(arm/"protocol.json")
    comparison=read(arm/"comparison.json")
    change=read(arm/"unlabelled_changes.json")
    require(source_identity(ROOT)==plan["runtime_sha256"] and sha(features)==plan["features_sha256"]
            and seal["independent_reload"] and seal["api"]==0 and comparison["input_seal"]==sha(arm/"sealed.json")
            and not comparison["designated_labels_opened"] and comparison["carried_accuracy"] is None
            and change["comparison"]==sha(arm/"comparison.json")
            and sha(arm/"reloaded.csv")==seal["prediction_sha256"],"Research delivery is not sealed as unscored")
    predictions(arm/"reloaded.csv",change["rows"])
    activate_resource_directory(root/"discovery-library")
    output=root/"handoff-recruitment-unscored"
    output.mkdir(exist_ok=True)
    with TemporaryDirectory(dir=output) as temp:
        stage=Path(temp)
        titles=stage/"titles"
        titles.mkdir()
        for name in ("run","physical.jsonl","delivery"):
            (titles/name).symlink_to(arm/name)
        (titles/"features.zip").symlink_to(features)
        save(stage/"protocol.json",dict(original_run=plan,packages={"titles":{"features":sha(features)}},
            seal=sha(arm/"sealed.json"),unlabelled_change=change,score_comparison=comparison))
        save(stage/"validation.json",dict(status="sealed_predictions_without_designated_score",rows=change["rows"],
            accuracy=None,correct=None,designated_labels_opened=False,official_test_accessed=False,
            runtime_sha256=plan["runtime_sha256"],model_sha256=seal["model_sha256"],prediction_sha256=seal["prediction_sha256"],
            independent_reload=True,source=plan["arm"],fits=seal["fits"]))
        package(SimpleNamespace(run=stage,notices=root/"notices",output=output))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("root","features"):
        p.add_argument("--"+name,type=lambda s:Path(s).resolve(),required=True)
    a=p.parse_args()
    run(a.root,a.features)
