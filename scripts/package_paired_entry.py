"""Package the actual matched-finalist delivery using its original runtime and frozen inputs."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from scripts.cache_paired_encoders import configure_weights
from scripts.unified_released_study import read, require, save, sha
from tools.package_verified_delivery import package


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("raw","features","weights","notices","output"):
        parser.add_argument("--"+name,required=True,type=lambda s:Path(s).resolve())
    a=parser.parse_args()
    arm=a.raw/"race"
    sealed=read(arm/"sealed.json")
    measured=read(a.raw/"validation.json")["race"]
    run=read(arm/"protocol.json")
    require(sealed["independent_reload"] and sha(a.features)==run["features_sha256"] and
            read(a.raw/"evaluation_lock.json")["seals"]["race"]==sha(arm/"sealed.json"),
            "Scored model or input no longer matches the original freeze")
    configure_weights(a.weights)
    a.output.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=a.output) as tmp:
        staging=Path(tmp)
        titles=staging/"titles"
        titles.mkdir()
        for name in ("delivery","physical.jsonl","run"):
            (titles/name).symlink_to(arm/name)
        (titles/"features.zip").symlink_to(a.features)
        save(staging/"validation.json",dict(**measured,runtime_sha256=run["runtime_sha256"],
             prediction_sha256=sealed["prediction_sha256"],model_sha256=sealed["model_sha256"],
             independent_reload=True,source="matched-finalists",fits=sealed["fits"],official_test_accessed=False))
        save(staging/"protocol.json",dict(packages={"titles":{"features":sha(a.features)}},original_run=run,
             score_lock=read(a.raw/"evaluation_lock.json")))
        package(SimpleNamespace(run=staging,notices=a.notices,output=a.output))
