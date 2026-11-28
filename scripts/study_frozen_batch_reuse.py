"""Measure the real frozen encoder work required to obtain a trial's labels and probabilities."""
import argparse
import hashlib
import os
from pathlib import Path
import time
from unittest.mock import patch

import joblib
import numpy as np

from scripts.compare_regrouped_portfolios import lock
from scripts.unified_released_study import ROOT, read, require, sha
from superagent import ab_sentence
from superagent.external_assets import activate_resource_directory


PLAN = ROOT/"research/paired_encoder_search_1127/frozen_reuse_protocol.json"


def digest(array):
    value = np.ascontiguousarray(array)
    return hashlib.sha256(value.dtype.str.encode()+str(value.shape).encode()+value.tobytes()).hexdigest()


def run(root):
    out = root/"frozen-reuse"
    out.mkdir(exist_ok=False)
    agent = root/"analogy-portfolio-raw/portfolio_revised/run/agent"
    selection = read(agent/"analogy_review.json")
    member = selection["source_member"]
    trial = next(r for r in read(agent/"history.json")
                 if r["candidate"] == member and r["split_id"] == "primary" and r["status"] == "completed")
    file = agent/f"trial_{trial['trial']}.joblib"
    payload = agent/"primary.joblib"
    require(sha(payload)==read(agent/"primary_split.json")["payload_sha256"], "Changed training rows")
    model = joblib.load(file)["model"]
    training = joblib.load(payload)
    fit, held = training["X_train"], training["X_search"]
    require(len(fit) >= 400 and len(held) >= 192, "Incomplete training folds")
    activate_resource_directory(root/"discovery-library")
    os.environ.pop("SUPERAGENT_PAIRED_FEATURE_CACHE",None)
    lock(out/"inputs.json",dict(protocol=sha(PLAN),component=member,trial=sha(file),payload=sha(payload),
                                election=sha(agent/"analogy_review.json")))
    rows = []
    for size, order in ((96,(False,True)),(192,(True,False))):
        reference = None
        for online in order:
            os.environ["SUPERAGENT_FROZEN_BATCH_REUSE"] = str(int(online))
            model.predict(held[-5:])
            ab_sentence._recent_batch.cache_clear()
            with patch.object(ab_sentence,"encode_frozen",wraps=ab_sentence.encode_frozen) as observed:
                wall = time.perf_counter()
                cpu = time.process_time()
                a = model.predict(held[:size])
                b = model.predict(fit[:2*size])
                p = model.predict_proba(held[:size])
                elapsed_cpu, elapsed_wall = time.process_time()-cpu, time.perf_counter()-wall
            outputs = [digest(x) for x in (a,b,p)]
            if reference is None:
                reference = outputs
            require(outputs==reference,"Cached and uncached outputs differ")
            result = dict(held_rows=size,fitting_rows=2*size,reuse=online,actual_encoder_calls=observed.call_count,
                          cpu_seconds=elapsed_cpu,wall_seconds=elapsed_wall,outputs=outputs)
            lock(out/f"n{size}_reuse{int(online)}.json",result)
            rows.append(result)
            print(size,online,observed.call_count,round(elapsed_cpu,2),round(elapsed_wall,2),flush=True)
    comparisons=[]
    for n in (96,192):
        d={r["reuse"]:r for r in rows if r["held_rows"]==n}
        comparisons.append(dict(held_rows=n,identical=d[False]["outputs"]==d[True]["outputs"],
            saved_calls=d[False]["actual_encoder_calls"]-d[True]["actual_encoder_calls"],
            cpu_gain=1-d[True]["cpu_seconds"]/d[False]["cpu_seconds"],
            wall_gain=1-d[True]["wall_seconds"]/d[False]["wall_seconds"]))
    passed=all(r["identical"] and r["saved_calls"]>=1 and r["cpu_gain"]>0 for r in comparisons)
    lock(out/"results.json",dict(inputs=sha(out/"inputs.json"),rows=rows,comparisons=comparisons,passed=passed,
         supervised_fits=0,model_api_requests=0,designated_validation_opened=False,official_test_opened=False,H2_opened=False))
    print(comparisons,passed,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=lambda v:Path(v).resolve(),required=True)
    run(parser.parse_args().root)
