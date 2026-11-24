"""Freeze lexical overlap without labels, then measure the two actual agents on the same original rows."""
import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import re
import tempfile
import unicodedata

import numpy as np

from scripts.run_conditional_handoff import comparison
from scripts.study_brier_replication import RULE, partitions
from scripts.unified_released_study import ROOT, read, require, save, sha
from scripts.verify_brier_replication import ORIGINAL, PUBLIC as REGROUPED, trials_on_original_rows
from scripts.verify_crossfit_challenger import checksums, seal_directory
from scripts.verify_lifecycle_audit import copy

PLAN = ROOT / "docs/batches/pilot_title_overlap_plan_20261124_zh.md"
PUBLIC = ROOT / "research/pilot_title_overlap_1124"


def shingles(title):
    tokens = re.findall(r"[^\W_]+", unicodedata.normalize("NFKC", title).casefold())
    features = set(zip(tokens, tokens[1:]))
    return features if len(tokens) >= 6 and len(features) >= 5 else set()


def matching_pairs(feature_sets):
    postings = defaultdict(list)
    for i, current in enumerate(feature_sets):
        intersections = Counter(j for term in current for j in postings[term])
        for j, count in intersections.items():
            if 9 * count >= 4 * (len(current) + len(feature_sets[j])):
                yield j, i
        for term in current:
            postings[term].append(i)


def linked_groups(rows, groups):
    title_group = {}
    for pair, group in zip(rows, groups):
        for title in pair:
            require(title_group.setdefault(title, group) == group, "Exact title spans original groups")
    titles = sorted(title_group)
    features = [shingles(t) for t in titles]
    neighbors = defaultdict(set)
    parents = {g: g for g in groups}
    def root(g):
        while parents[g] != g:
            parents[g] = parents[parents[g]]
            g = parents[g]
        return g
    edges = set()
    matched_titles = 0
    for j, i in matching_pairs(features):
        a, b = title_group[titles[j]], title_group[titles[i]]
        if a == b:
            continue
        matched_titles += 1
        neighbors[titles[i]].add(a)
        neighbors[titles[j]].add(b)
        edges.add(tuple(sorted((a, b))))
        first, second = root(a), root(b)
        parents[max(first, second)] = min(first, second)
    expanded = [root(g) for g in groups]
    diagnostics = dict(unique_titles=len(titles), eligible_titles=sum(bool(x) for x in features),
                       title_pairs_between_original_groups=matched_titles, edges_between_original_groups=len(edges),
                       original_groups=len(parents), expanded_groups=len(set(expanded)),
                       largest_expanded_rows=max(Counter(expanded).values()))
    return neighbors, expanded, diagnostics


def membership(id_sets, main):
    folds = {int(row): i for i, (_, held) in enumerate(id_sets) for row in held}
    require(len(folds) == len(main) and set(folds) == set(main), "Fold assignment changed")
    return [folds[int(i)] for i in main]


def compute(rows, groups, main, assignments):
    neighbors, expanded, diagnostics = linked_groups(rows, groups)
    result = [dict(row_id=int(main[i]), expanded_group=expanded[i]) for i in range(len(main))]
    for name, folds in assignments.items():
        original_fold = {}
        cluster_folds = defaultdict(set)
        for g, cluster, fold in zip(groups, expanded, folds):
            require(original_fold.setdefault(g, fold) == fold, "Original group split")
            cluster_folds[cluster].add(fold)
        for i, (pair, fold, cluster) in enumerate(zip(rows, folds, expanded)):
            related = neighbors[pair[0]] | neighbors[pair[1]]
            direct = any(original_fold[g] != fold for g in related)
            intact = len(cluster_folds[cluster]) == 1
            require(not (direct and intact), "Inconsistent overlap definitions")
            result[i].update({name + "_fold": fold, name + "_direct": int(direct), name + "_intact": int(intact)})
    return result, diagnostics


def freeze(folder, archive):
    task, groups, fresh, old = partitions(archive)
    main = np.sort(np.concatenate([held for _, held in old]))
    rows = [task.X_train[i] for i in main]
    assignments = {k: membership(s, main) for k, s in (("first", old), ("second", fresh))}
    records, info = compute(rows, groups[main].tolist(), main, assignments)
    folder.mkdir(parents=True, exist_ok=False)
    table = folder / "flags.csv"
    with table.open("x", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    save(folder / "frozen.json", dict(plan=sha(PLAN), source=sha(Path(__file__)), flags=sha(table),
                                       rows=len(records), diagnostics=info, classifier_fits=0, model_api_calls=0))
    print("Frozen overlap groups and row flags:", info)


def score(folder, archive):
    frozen = read(folder / "frozen.json")
    require(frozen["plan"] == sha(PLAN) and frozen["flags"] == sha(folder / "flags.csv")
            and frozen["source"] == sha(Path(__file__)), "Overlap definition or labels changed")
    with (folder / "flags.csv").open(encoding="utf-8") as stream:
        flags = list(csv.DictReader(stream))
    ids = np.array([int(r["row_id"]) for r in flags])
    expanded = np.array([r["expanded_group"] for r in flags])
    task, _, fresh, old = partitions(archive)
    y = np.asarray(task.y_train)
    output = {}
    for name, path, sets, key, arm_summary in (("first", ORIGINAL, old, RULE + "_approved", "arms"),
                                              ("second", REGROUPED, fresh, "agent", "performance")):
        p = trials_on_original_rows(path, sets, key, len(y))[ids]
        require(np.all(p >= 0), "Missing earlier prediction")
        direct = np.array([r[name + "_direct"] == "1" for r in flags])
        intact = np.array([r[name + "_intact"] == "1" for r in flags])
        subsets = {"all": np.ones(len(ids), dtype=bool), "detected_direct_crossing": direct,
                   "no_detected_direct_crossing": ~direct, "expanded_group_intact": intact}
        metrics = {}
        for label, mask in subsets.items():
            metrics[label] = comparison(p[mask, 0], p[mask, 1], y[ids[mask]], expanded[mask]) if mask.any() else None
        summary = read(path / "result.json")[arm_summary]
        if name == "first":
            summary = summary[key]
        full = metrics["all"]
        require(all(full[k] == summary[k] for k in ("rows", "base_correct", "correct", "gained", "lost", "net")),
                "Published aggregate did not match the same original rows")
        output[name] = dict(parent=sha(path / "result.json"), scores=metrics,
                            direct_crossing_rows=int(direct.sum()), expanded_crossing_rows=int((~intact).sum()))
    save(folder / "result.json", dict(frozen=sha(folder / "frozen.json"), arms=output))
    for name, arm in output.items():
        print(name, {k: (v["rows"], v["net"]) if v else None for k, v in arm["scores"].items()})


def verify(archive):
    require(checksums(PUBLIC) == (PUBLIC / "SHA256SUMS").read_text(), "Evidence changed")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "check"
        freeze(out, archive)
        score(out, archive)
        for name in ("flags.csv", "frozen.json", "result.json"):
            require(sha(out / name) == sha(PUBLIC / name), "Recomputed data differ: " + name)
    print("Verified text-only overlap flags and both paired comparisons.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="stage", required=True)
    for name in ("freeze", "score", "publish", "verify"):
        command = subs.add_parser(name)
        if name != "verify":
            command.add_argument("out", type=Path)
        if name != "publish":
            command.add_argument("archive", type=Path)
    args = parser.parse_args()
    if args.stage == "freeze":
        freeze(args.out, args.archive)
    elif args.stage == "score":
        score(args.out, args.archive)
    elif args.stage == "publish":
        for name in ("flags.csv", "frozen.json", "result.json"):
            copy(args.out / name, PUBLIC / name)
        seal_directory(PUBLIC)
    else:
        verify(args.archive)
