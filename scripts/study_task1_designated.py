"""One isolated designated grade for an independently qualified ordinary training policy."""
import argparse
import io
from pathlib import Path
import zipfile

import numpy as np

from scripts.study_task1_outer import prediction_csv
from scripts.unified_released_study import read,require,save,sha


def grade(archive,out):
    match=out/"fusion_match/result.json"
    require(read(match)["advance"],"No policy has qualified on the new training folds")
    predictions={}
    locks={}
    for arm in ("anchor","release"):
        folder=out/arm
        record=read(folder/"result.json")
        actual=sha(folder/"solve.csv")
        require(actual==record["local_prediction_sha256"]==sha(folder/"replay.csv") and record["replayed"] and
                record["model_api_requests"]==0 and not record["designated_labels_opened"],"Unverified designated prediction")
        predictions[arm]=prediction_csv(folder/"solve.csv",2030)
        locks[arm]=dict(record=sha(folder/"result.json"),predictions=actual)
    save(out/"designated_lock.json",dict(qualification=sha(match),archive=sha(archive),solutions=locks))
    with zipfile.ZipFile(archive) as f:
        item,=[n for n in f.namelist() if Path(n).name=="y_val.npy" and "__MACOSX" not in n]
        y=np.load(io.BytesIO(f.read(item)),allow_pickle=False).astype(str).reshape(-1)
    require(len(y)==2030,"Unexpected designated label count")
    a,b=[predictions[arm]==y for arm in ("anchor","release")]
    result=dict(lock=sha(out/"designated_lock.json"),rows=len(y),local_anchor_correct=int(a.sum()),
                release_correct=int(b.sum()),local_anchor_accuracy=float(a.mean()),release_accuracy=float(b.mean()),
                fixed=int((b & ~a).sum()),lost=int((a & ~b).sum()),historical_partner_correct=1549,
                official_test_opened=False,model_api_requests=0,designated_grade_used_for_selection=False)
    save(out/"designated_result.json",result)
    print(result,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ("archive","out"):
        parser.add_argument("--"+k,required=True,type=lambda s:Path(s).resolve())
    a=parser.parse_args()
    grade(a.archive,a.out)
