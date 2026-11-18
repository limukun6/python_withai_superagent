"""Inspect the first score-sensitive action in saved 4/3 trajectories."""
import argparse
from collections import Counter
import json
from pathlib import Path

from superagent.evidence_agent import route_of
from superagent.local_search import LocalSearchPolicy
from superagent.models import search_candidates


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def inspect(case, kind, seed):
    control = read(case)["4_3"]
    folder = Path(control["model_file"]).parent
    history = read(folder / "trials.json")
    catalog = [dict(id=c.family, params=c.params, route=route_of(c.family))
               for c in search_candidates(kind, "expanded")]
    prefix = [dict(candidate=r["candidate"], status=r["status"], search_score=r["search_score"],
                   route=route_of(r["candidate"])) for r in history[:2]]
    if (len(history) != 4 or any(r["status"] != "completed" for r in history)
            or prefix[0]["candidate"] != catalog[0]["id"]):
        raise ValueError(f"Expected completed four-step anchored trajectory in {folder}")
    actions = {}
    scores = [r["search_score"] for r in prefix]
    for name, values in [("observed", scores), ("constant", [0.5, 0.5]), ("swapped", scores[::-1])]:
        policy = LocalSearchPolicy("interleaved_routes", catalog, seed)
        if policy.choose(prefix[:1])[0] != prefix[1]["candidate"]:
            raise ValueError(f"Second action differs in {folder}")
        visible = [dict(r, search_score=value) for r, value in zip(prefix, values)]
        actions[name] = policy.choose(visible)[0]
    if actions["observed"] != history[2]["candidate"]:
        raise ValueError(f"Observed-score replay differs in {folder}")
    confirmation = read(folder / "confirmation.json")
    rank = confirmation["shortlist"].index(control["candidate"]) + 1
    return dict(task=case.parts[-3], outer_seed=int(case.parts[-2]), actions=actions, winner_search_rank=rank)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, required=True)
    args = parser.parse_args()
    protocol = read(args.runs / "expanded_feedback_pilot" / "pilot_protocol.json")
    cases = []
    for study in ("search_budget_study_v1", "search_budget_replication_v1"):
        for path in sorted((args.runs / study).glob("*/*/frozen_arms.json")):
            kind = protocol["tasks"][path.parts[-3]]["kind"]
            cases.append(inspect(path, kind, protocol["search_seed"]))
    pairs = {(r["task"], r["outer_seed"]) for r in cases}
    expected = {(task, seed) for task in protocol["tasks"] for seed in (42011, 42017, 42041, 42053, 42067)}
    if pairs != expected or len(cases) != len(expected):
        raise ValueError("The two studies must contain one baseline for each registered task and seed")
    print(json.dumps(dict(cases=cases, winner_ranks=Counter(r["winner_search_rank"] for r in cases)),
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
