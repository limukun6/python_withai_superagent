"""Check actual pinned weights in isolated directories, without fitting a model."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from scripts.unified_released_study import save

SHARED = ("all-MiniLM-L12-v2","all-MiniLM-L6-v2","e5-small-v2")
EXTRA = ("bge-small-en-v1.5","all-mpnet-base-v2","roberta-base")
CASES = {"complete":6,"optional_missing":5,"optional_incomplete":5,"reference_missing":0}
PROBE = """
import json, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scripts import unified_encoder_portfolio as entry
weights = Path(sys.argv[1])
entry.configure(weights)
task = SimpleNamespace(kind="pair_text", y_train=np.array([0,1]),
    spec={"metric":"accuracy","input_schema":{"pair_semantics":"preference","pair_label_one":"left"}})
try:
    if hasattr(entry, "candidate_inventory"):
        record = entry.candidate_inventory(task,weights)
    else:
        record = {"candidates":entry.candidates(task,weights)}
except ValueError as error:
    record = {"error":str(error)}
print(json.dumps(record))
"""


def probe(source, weights):
    results = {}
    for case,target in CASES.items():
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in (*SHARED,*EXTRA):
                if name == EXTRA[0] and case in ("optional_missing","optional_incomplete"):
                    if case == "optional_incomplete":
                        (root/name).mkdir()
                    continue
                if name == SHARED[1] and case == "reference_missing":
                    continue
                (root/name).symlink_to((weights/name).resolve(),target_is_directory=True)
            env = dict(os.environ, AB_B1_OFFLINE="1", PYTHONPATH=os.pathsep.join(map(str,(source/"scripts/ab_acceptance_guard",source))))
            observed = json.loads(subprocess.check_output([sys.executable,"-c",PROBE,str(root)],
                                  cwd=source, env=env, text=True))
            names = observed.get("candidates",[])
            passed = "error" not in observed and len(names)==target and (case not in ("optional_missing","optional_incomplete") or EXTRA[0] not in names)
            results[case] = dict(expected=target,passed=passed,**observed)
    return results


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("source","weights","output"):
        p.add_argument(name,type=lambda x:Path(x).resolve())
    a=p.parse_args()
    save(a.output,probe(a.source,a.weights))
