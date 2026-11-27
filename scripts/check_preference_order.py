"""Check graph feasibility using the fitting side of an existing grouped training payload."""
import argparse
from pathlib import Path

import joblib

from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent.preference_order import inspect_order

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=lambda s:Path(s).resolve(),required=True)
    root=p.parse_args().root
    source=root/"raw/race/run/agent"
    payload=source/"primary.joblib"
    require(sha(payload)==read(source/"primary_split.json")["payload_sha256"],"Training partition changed")
    data=joblib.load(payload)
    require(data.get("pair_semantics")=="preference","Order has not been declared")
    evidence, _=inspect_order(data["X_train"],data["y_train"])
    out=root/"order-feasibility"
    out.mkdir(exist_ok=True)
    result=dict(protocol_sha256=sha(ROOT/"research/paired_encoder_search_1127/order_graph_protocol.json"),
                training_payload_sha256=sha(payload),evidence=evidence, actual_classifier_fits=0, new_model_api_requests=0,
                heldout_labels_used=False, released_validation_opened=False, formal_test_opened=False, H2_opened=False)
    if (out/"results.json").exists():
        require(result==read(out/"results.json"),"Prior result is bound to different data")
    else:
        save(out/"results.json",result)
    print(evidence, flush=True)
