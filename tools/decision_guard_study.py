"""Replay twelve labeled decision cases through the current offline checks."""
import argparse
import csv
import io
import json
from pathlib import Path
import zipfile

from superagent.adaptive_local import save
from superagent.configuration_facts import configuration_change
from superagent.contracts import ExperimentProposal
from superagent.hypothesis_agent import check_expectation, parse_proposal
from superagent.hypothesis_memory import build_memory
from superagent.schema import load_manifest_split, read_manifest


def text_pair_kind(semantics, declared_label=None):
    manifest = dict(schema_version=1, modality="pair_text", columns=[dict(name="left", type="text"), dict(name="right", type="text")],
                    target="label", pair_semantics=semantics,
                    splits={"train": {"format": "csv", "path": "train.csv"}})
    if declared_label is not None:
        manifest["pair_label_one"] = declared_label
    content = io.StringIO()
    writer = csv.writer(content)
    writer.writerows([["left", "right", "label"], ["one", "two", 1], ["red", "blue", 0]])
    with zipfile.ZipFile(io.BytesIO(), "w") as archive:
        archive.writestr("train.csv", content.getvalue())
        archive.writestr("manifest.json", json.dumps(manifest))
        return load_manifest_split(archive, read_manifest(archive), "train")[0]


BASE = {"trial": 1, "candidate": "word_char_C1.5", "params": {"C": 1.5},
        "status": "completed", "search_accuracy": .6964, "generalization_gap": .10}
NEXT = {"id": "word_char_C0.05", "params": {"C": .05}}
PROPOSAL = {"candidate": NEXT["id"], "reason": "Try stronger regularization",
            "hypothesis": "Strong regularization may improve search accuracy", "review": "Evidence is limited",
            "evidence_trials": [1], "expected_change": {"metric": "search_accuracy", "reference_trial": 1,
                                                         "direction": "increase", "min_delta": .001}}


def configuration_valid(params):
    proposed = ExperimentProposal.from_decision(2, dict(id=NEXT["id"], params=params), "Check C", PROPOSAL, [BASE])
    proposed.validate_available([NEXT], [BASE])
    return True


def evidence_valid(reference):
    parse_proposal(PROPOSAL | dict(evidence_trials=[reference]), [BASE], [NEXT])
    return True


def outcome(score):
    return dict(trial=2, candidate=NEXT["id"], status="completed", search_accuracy=score, generalization_gap=.04)


def target_improved(score):
    return check_expectation(PROPOSAL, outcome(score), [BASE])["status"] == "supported"


def sole_parameter_changed(params):
    change = configuration_change(BASE, {"id": NEXT["id"], "params": params}, "explicit_reference")
    return change["single_declared_parameter_change"]


def memory_contains_claim(score):
    row = outcome(score)
    row.update(proposal=PROPOSAL, hypothesis_check=check_expectation(PROPOSAL, row, [BASE]))
    memory = build_memory([BASE, row], [])
    hypothesis = memory["hypotheses"][0]
    return hypothesis["status"] == "measurement_supported" and not hypothesis["causal_explanation_verified"]


def evaluate():
    cases = [
        ("task_semantics", "declared left-wins ranking", True, lambda: text_pair_kind("preference", "left") == "pair_text"),
        ("task_semantics", "reuse ranking augmentation for relation", False, lambda: text_pair_kind("relation") == "pair_text"),
        ("tool_config", "catalog C", True, lambda: configuration_valid({"C": .05})),
        ("tool_config", "additional unregistered parameter", False, lambda: configuration_valid({"C": .05, "magic": 1})),
        ("evidence_reference", "completed trial 1", True, lambda: evidence_valid(1)),
        ("evidence_reference", "nonexistent trial 9", False, lambda: evidence_valid(9)),
        ("primary_metric", "gap lower and accuracy above target", True, lambda: target_improved(.705)),
        ("primary_metric", "gap lower but accuracy falls to .6861", False, lambda: target_improved(.6861)),
        ("attribution", "record one controlled parameter difference", True, lambda: sole_parameter_changed({"C": .05})),
        ("attribution", "attribute two parameter changes to C alone", False,
         lambda: sole_parameter_changed({"C": .05, "representation": "char"})),
        ("memory", "new measurement supports the stated directional expectation", True, lambda: memory_contains_claim(.705)),
        ("memory", "keep a contradicted expectation as supported", False, lambda: memory_contains_claim(.6861)),
    ]
    records = []
    for category, case, expected, check in cases:
        detail = None
        try:
            observed = bool(check())
        except ValueError as error:
            observed, detail = False, str(error)
        records.append(dict(category=category, case=case, expected=expected,
                            id_format_only=True, current_check=observed, detail=detail))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = evaluate()
    stats = {}
    for arm in ("id_format_only", "current_check"):
        stats[arm] = dict(invalid_flagged=sum(not r["expected"] and not r[arm] for r in records),
                          invalid_total=sum(not r["expected"] for r in records),
                          valid_rejected=sum(r["expected"] and not r[arm] for r in records),
                          valid_total=sum(r["expected"] for r in records))
    if args.output.exists():
        raise FileExistsError(args.output)
    save(args.output, dict(cases=records, totals=stats, api_calls=0,
         scope="Unit-level decision signals, some advisory rather than runtime vetoes. "
               "Gap example uses historical search accuracies and illustrative gap values."))
    print(stats, flush=True)
    return int(any(row["expected"] != row["current_check"] for row in records))


if __name__ == "__main__":
    raise SystemExit(main())
