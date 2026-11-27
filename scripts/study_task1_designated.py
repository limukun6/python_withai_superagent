"""One isolated designated grade for an independently qualified ordinary training policy."""
import argparse
import io
from pathlib import Path
import zipfile

import numpy as np

from scripts.study_task1_outer import prediction_csv
from scripts.unified_released_study import read,require,save,sha


def grade(archive,out,study="fusion_match"):
    match=out/study/"result.json"
    budget = 32
    if study=="remaining_folds":
        require(read(match)["advance"] is True,"Additional held folds did not qualify")
        qualified,required_mode,indices,budget="extended","remaining-folds",(0,1,2),48
    elif study=="near_full":
        require(read(match)["advance"] is True,"Near-final-size confirmation did not qualify")
        qualified,required_mode,indices="near_full","near-full-confirmation",(3,4)
    else:
        qualified=read(match)["advance"]
        require(qualified in ("regular","separate"),"No policy has qualified on the new training folds")
        required_mode="fusion-holdout" if qualified=="separate" else None
        indices=(0,1,2)
    require(read(out/"release/inputs.json")["selection_policy"]==required_mode and read(out/"release/inputs.json")["fit_cap"]==budget,
            "Release differs from the policy that qualified")
    source_ids={read(out/study/str(i)/qualified/"launch.json")["source"] for i in indices}
    require(source_ids=={read(out/"release/inputs.json")["source"]},"Release is not the qualified runtime")
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
    parser.add_argument("--study",choices=("fusion_match","near_full","remaining_folds"),default="fusion_match")
    a=parser.parse_args()
    grade(a.archive,a.out,a.study)
