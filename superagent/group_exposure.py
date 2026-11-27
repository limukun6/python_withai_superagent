"""Generate supported training-weight experiments from observed errors on connected groups."""
from functools import partial
import hashlib
import json
import math

import numpy as np

from .data import TrainingTask
from .models import Candidate
from .paired_encoder import NAMES
from .splits import sample_groups

OPERATOR = "reweight_connected_groups"


def sizes(rows):
    groups = sample_groups(TrainingTask("pair_text", rows, None), "shared_text")
    _, positions, counts = np.unique(groups, return_inverse=True, return_counts=True)
    return counts[positions]


def errors(group_sizes, truth, predictions):
    right = np.asarray(truth) == np.asarray(predictions)
    if right.ndim != 1 or right.shape != group_sizes.shape:
        raise ValueError("Connected-group errors need row-aligned observations")
    bins = [right[group_sizes == 1], right[group_sizes > 1]]
    if min(map(len, bins)) < 30:
        return None
    risks = [1 - float(cell.mean()) for cell in bins]
    se = math.sqrt(sum(p*(1-p)/len(cell) for p, cell in zip(risks, bins)))
    gap = risks[1] - risks[0]
    return dict(isolated=len(bins[0]), repeated=len(bins[1]), isolated_error=risks[0], repeated_error=risks[1],
                gap=gap, standard_error=se, distinguishes=abs(gap) > se)


def identifier(spec):
    packed = json.dumps(spec, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return "group_weight_" + hashlib.sha256(packed).hexdigest()[:20]


def proposals(history, baseline):
    ranked = []
    for row in history:
        name = row["candidate"]
        evidence = row.get("connected_group_errors") or {}
        if (name in baseline and NAMES.get(name, (None, None))[1] == "linear" and row["split_id"] == "primary"
                and row["status"] == "completed" and evidence.get("distinguishes")):
            precision = abs(evidence["gap"]) / max(evidence["standard_error"], 1e-12)
            ranked.append((-row["search_score"], -precision, name, row))
    result = {}
    for _, _, name, row in sorted(ranked, key=lambda x: x[:3])[:2]:
        direction = .5 if row["connected_group_errors"]["gap"] > 0 else -.5
        for exponent in (direction, -direction):
            spec = dict(kind="pair_text", operator=OPERATOR, template=name, power=exponent)
            result[identifier(spec)] = spec
    return result


def materialize(spec):
    from .models import search_candidates
    if (set(spec) != {"kind", "operator", "template", "power"} or spec["kind"] != "pair_text"
            or spec["operator"] != OPERATOR or spec["power"] not in (-.5, .5)
            or NAMES.get(spec["template"], (None, None))[1] != "linear"):
        raise ValueError("Unsupported group-weight proposal")
    sources = search_candidates("pair_text", "all")
    base = next((item for item in sources if item.family == spec["template"]), None)
    if base is None:
        raise ValueError("Missing proposed encoder")
    params = dict(base.params, parent_template=base.family, connected_group_power=spec["power"])
    return Candidate(identifier(spec), params, partial(base.fit_predict, group_power=spec["power"]))


def capability(spec, baseline):
    base = baseline[spec["template"]]
    name = identifier(spec)
    params = dict(base["params"], parent_template=spec["template"], connected_group_power=spec["power"])
    return dict(base, id=name, params=params, origin="runtime_composed")
