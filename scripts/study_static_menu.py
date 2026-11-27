"""Run the anonymous degree-of-freedom comparison with one additional encoder family."""
import argparse
import os
from pathlib import Path

import joblib
import numpy as np

from scripts.compare_regrouped_portfolios import lock
from scripts.run_planner_self_review import source_identity
from scripts.study_portfolio_recruitment import score, solve
from scripts.study_resource_menu import inventory
from scripts.unified_released_study import ROOT, read, require, sha
from superagent.external_assets import activate_resource_directory
from superagent.local_sentence import encode

PLAN = ROOT/"research/paired_encoder_search_1127/static_embedding_protocol.json"


def prepare(root):
    from sentence_transformers import SentenceTransformer, __version__
    require(__version__=="3.4.1","Run the comparison with the registered test-only reference")
    out = root/"static-menu-cold"
    case = out/"comparison"
    case.mkdir(parents=True,exist_ok=True)
    previous = root/"portfolio-recruitment"
    base = read(previous/"inputs.json")
    old = inventory(root/"arctic-library")
    library = root/"potion-library"
    new = inventory(library)
    require(old.keys() <= new.keys() and len(new)-len(old)==1 and sorted(old)==base["resources"]
            and all(old[s]==new[s] for s in old),"More than the proposed resource changed")
    added = next(iter(new.keys()-old.keys()))
    spec=new[added]
    require(spec["family"]=="static_embedding","Expected the additional family")
    activate_resource_directory(library)
    ref = SentenceTransformer(os.environ[spec["environment_variable"]],device="cpu",local_files_only=True)
    training = joblib.load(previous/"comparison/challenger/run/agent/refit_data.joblib")
    examples = list(dict.fromkeys(s for pair in training["X_train"] for s in pair))[:96]
    examples += ["","  ","?!.,","東京很美。","MiXeD-case word_123","hello hello repeated",
                 "[UNK] 🙂 café","multiple   spaces\tbetween\nwords","a test "*2000]
    a = ref.encode(examples,show_progress_bar=False)
    ours = encode(spec,examples)
    maximum = float(np.abs(a-ours).max())
    require(ours.shape==a.shape and np.isfinite(ours).all() and maximum<=.000002,"Static encoder differs from reference")
    lock(out/"parity.json",dict(protocol=sha(PLAN),source=source_identity(ROOT),reference=__version__,rows=len(a),
        width=ours.shape[1],max_abs=maximum,added=added,existing_descriptors_equal=len(old),
        installation=sha(root/"potion-installed.json")))
    for name in ("p.zip","q.zip","truth.joblib","fit_features"):
        target=case/name
        if not target.exists():
            target.symlink_to(previous/"comparison"/name)
    control=case/"control"
    if not control.exists():
        control.symlink_to(previous/"comparison/challenger")
    plan=dict(base, control=base["challenger"],challenger=source_identity(ROOT),resources=sorted(new),
              plan=sha(PLAN),prior=sha(previous/"comparison/challenger/sealed.json"),
              candidates_added=[added],library_parity=sha(out/"parity.json"))
    lock(out/"inputs.json",plan)
    print(len(a),maximum,added,flush=True)


def trial(root):
    out=root/"static-menu-cold"
    solve(root,ROOT,"comparison","challenger",out=out,library=root/"potion-library")
    score(root,"comparison",out=out)
    notes=read(out/"inputs.json")
    name="local_sentence_"+notes["candidates_added"][0][:20]
    directory=out/"comparison/challenger/run/agent"
    history=read(directory/"history.json")
    done=read(directory/"run_outcome.json")
    decision=read(directory/"portfolio_review.json")
    observed=[dict(split=r.get("split_id"),status=r["status"]) for r in history if r.get("candidate")==name]
    lock(out/"choice.json",dict(result=sha(out/"comparison/results.json"),candidate=name,trial_history=observed,
         included_in_primary_reconsideration=any(r["member"]==name for r in decision.get("considered",[])),
         final=done["members"],in_final=name in done["members"],physical_fits=done["attempted_classifier_fits"]))
    print(name,observed,done["members"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("prepare","trial"))
    p.add_argument("--root",required=True,type=lambda s:Path(s).resolve())
    args=p.parse_args()
    if args.action=="prepare":
        prepare(args.root)
    else:
        trial(args.root)
