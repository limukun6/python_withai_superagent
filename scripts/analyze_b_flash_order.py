"""Audit two published, fully observed SMS search trajectories without training."""
import argparse
import hashlib
import itertools
import json
from decimal import Decimal
from pathlib import Path
from statistics import mean
from subprocess import check_output

ROOT = Path(__file__).resolve().parents[1]
REVISION = "7dd5e6c719fa15ef29ccd9b599104e786d29c280"
PILOT = Path("research/a28_flash_network_resume/sms_pilot")
ARMS = ("raw-history", "checked-history")


def source_reader(hashes):
    def read(relative):
        path = PILOT / relative
        raw = check_output(["git", "show", f"{REVISION}:{path.as_posix()}"], cwd=ROOT)
        if (ROOT / path).read_bytes() != raw:
            raise ValueError(f"Working copy differs from the A28 archive: {path}")
        hashes[relative] = hashlib.sha256(raw).hexdigest()
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in raw.splitlines()]
        return json.loads(raw)
    return read


def audit_arm(name, read, protocol):
    def get(item):
        return read(f"{name}/{item}")
    manifest = get("run_manifest.json")
    history = get("history.json")
    events = get("events.jsonl")
    actions = get("actions.json")
    outcome = get("run_outcome.json")
    selected = get("final_selection.json")
    verification = get("verification.json")
    usage = [u for u in get("llm_usage.jsonl") if u["event"] == "llm_call"]
    routes = {e["trial"]: e for e in events if e["kind"] == "planner_routing"}
    admitted = [e for e in events if e["kind"] == "scheduler_action_admitted"]
    observed = [e for e in events if e["kind"] == "scheduler_action_observed"]
    proposals = {e["trial"]: e["proposal"] for e in events if e["kind"] == "ab_flash_proposal"}
    chosen = [h["candidate"] for h in history]
    primary = protocol["splits"]["primary"]
    if not (len(chosen) == len(set(chosen)) == 6 and set(chosen) == set(manifest["capabilities"])
            and manifest["capabilities"] == protocol["capabilities"]
            and manifest["task_contract"]["metric"]["name"] == "macro_f1"
            and manifest["seed"] == primary["seed"]
            and len(actions) == len(admitted) == len(observed) == 7 and actions[-1]["action"] == "stop"
            and outcome["status"] == "completed" and outcome["requested_planner_completed"]
            and not outcome["formal_test_accessed"] and not selected["test_accessed"]
            and selected["refit"] and verification["reload_identical"]
            and selected["model_sha256"] == verification["model_sha256"]
            and len(usage) == len(proposals) == outcome["new_api_requests"] == 3):
        raise ValueError(f"Published protocol, menu or receipts failed verification: {name}")
    rows, by_candidate = [], {}
    for i, hist in enumerate(history):
        action, admission, observation = actions[i], admitted[i], observed[i]
        step, candidate, score = hist["trial"], hist["candidate"], hist["search_score"]
        route = routes[step]
        proposal = proposals.get(step)
        if not (step == i + 1 and admission["action"] == action and action["action"] == route["action"] == "explore"
                and action["candidate_id"] == candidate and action["split_id"] == hist["split_id"] == "primary"
                and admission["candidate"]["params"] == hist["params"] and admission["split"] == primary
                and admission["state_sha256"] == observation["state_sha256"] and hist["status"] == "completed"
                and hist["primary_metric"] == observation["observation"]["primary_metric"] == "macro_f1"
                and observation["observation"]["search_score"] == score
                and ((route["source"] == "flash") == (proposal is not None) == (hist["proposal"] is not None))):
            raise ValueError(f"Action and measured score disagree: {name} trial {step}")
        if proposal and not (proposal["candidate"] == candidate and proposal["raw_proposal"] == hist["proposal"]):
            raise ValueError(f"Stored Flash proposal differs from the executed candidate: {step}")
        by_candidate[candidate] = score
        rows.append(dict(step=step, candidate=candidate, source=route["source"], observed=score,
                         best_seen=max(by_candidate.values())))
    best = max(by_candidate.values())
    if not (by_candidate[selected["single_candidate"]] == selected["selection_score"] == best
            and selected["single_candidate"] == verification["selected_candidate"]):
        raise ValueError(f"Unexpected final selection: {name}")
    for row in rows:
        row["gap_to_measured_best"] = best - row["best_seen"]
    return dict(prefix=rows, first_best=next(r["step"] for r in rows if r["best_seen"] == best),
                by_candidate=by_candidate, task_archive_sha256=manifest["task_archive_sha256"],
                primary_split=primary, final_verification=verification, original_fits=outcome["classifier_fit_accounting"],
                original_usage=dict(calls=len(usage), input_tokens=sum(u["input_tokens"] for u in usage),
                                    output_tokens=sum(u["output_tokens"] for u in usage),
                                    course_rmb=str(sum(Decimal(str(u["course_cost_rmb"])) for u in usage))))


def audit():
    hashes = {}
    read = source_reader(hashes)
    protocol = read("protocol.json")
    arms = {name: audit_arm(name, read, protocol) for name in ARMS}
    raw, checked = arms.values()
    for field in ("task_archive_sha256", "primary_split", "by_candidate", "final_verification"):
        if raw[field] != checked[field]:
            raise ValueError(f"Cross-arm mismatch in {field}")
    first = raw["prefix"][0]["candidate"]
    if first != checked["prefix"][0]["candidate"]:
        raise ValueError("The first action differs across arms")
    scores = raw["by_candidate"]
    rest = sorted(set(scores) - {first})
    orders = [(first, *tail) for tail in itertools.permutations(rest)]
    reference = []
    for k in range(1, len(scores) + 1):
        bests = [max(scores[c] for c in order[:k]) for order in orders]
        reference.append(dict(step=k, mean=mean(bests), minimum=min(bests), maximum=max(bests),
                              orders_reaching_best=sum(x == max(scores.values()) for x in bests)))
    return dict(source_commit=REVISION, source_sha256=hashes, arms=arms, registered_splits=protocol["splits"],
                fixed_first_candidate=first, possible_orders=len(orders), prefix_reference=reference,
                new_classifier_fits=0, new_model_requests=0, new_course_rmb=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
