"""Choose and corroborate competing single models and votes from observed training rows."""
from itertools import combinations

import numpy as np

from .conditional_review import compare, group_observation
from .experiment_policy import identity


def subsets(names):
    return [part for size in range(1, len(names)) for part in combinations(names, size)]


def observation(records, group, metric):
    if len(group) == 1 or metric.name != "accuracy":
        return group_observation(records, group, metric)
    members = [records[name] for name in group]
    if any(row["probabilities"] is None for row in members):
        return None
    first = members[0]
    if any(row["row_key"] != first["row_key"] or
           not np.array_equal(row["labels"], first["labels"]) for row in members[1:]):
        raise ValueError("Cannot vote on unmatched evaluation rows")
    probabilities = np.mean([row["probabilities"] for row in members], axis=0)
    predicted = np.asarray(metric.labels)[probabilities.argmax(axis=1)]
    return dict(first, predictions=predicted, probabilities=probabilities,
                score=float(np.mean(predicted == first["labels"])))


def shortlist(primary, metric, allow_votes, available):
    singles = sorted(((name,) for name in available), key=lambda g: (-primary[g[0]]["score"], identity(g)))
    if len(singles) < 3:
        return None
    anchor = singles[0]
    records = {g: primary[g[0]] for g in singles}
    def evaluate(group):
        if group not in records:
            records[group] = observation(primary, group, metric)
        return records[group]
    def improves_subsets(group):
        value = evaluate(group)
        controls = [evaluate(other) for other in subsets(group)]
        return (value is not None and all(row is not None for row in controls)
                and value["score"] > max(records[anchor]["score"], *(r["score"] for r in controls)))
    blends = []
    if allow_votes:
        names = sorted(available)
        pairs = [pair for pair in combinations(names, 2) if improves_subsets(pair)]
        pairs.sort(key=lambda g: (-records[g]["score"], identity(g)))
        # Only extend genuinely complementary pairs. Limit search multiplicity before confirmation.
        triples = {tuple(sorted((*pair, n))) for pair in pairs[:2] for n in names if n not in pair}
        blends = [*pairs, *(triple for triple in triples if improves_subsets(triple))]
    if blends:
        challenger = min(blends, key=lambda g: (-records[g]["score"], len(g), identity(g)))
    else:
        runner = singles[1]
        matched = compare("primary", records[anchor], records[runner])
        if not matched["gained"] or -matched["delta"] > 1/np.sqrt(matched["rows"]):
            return None
        challenger = runner
    groups = sorted({anchor, challenger, *subsets(challenger)}, key=lambda g:(len(g), identity(g)))
    return anchor, challenger, groups, compare("primary", records[anchor], records[challenger])


class FinalistRace:
    def __init__(self, logger, folds):
        self.logger = logger
        self.rows = {name: set(map(int, held)) for name, (_, held, _) in folds.items()}
        self.plan = None
        self.cancelled = False

    def consider(self, menu, caps, observed, metric, remaining, blocked, allow_votes):
        if self.cancelled:
            return None
        primary = observed["primary"]
        if self.plan is None:
            fresh = []
            occupied = set(self.rows["primary"])
            for name, held in self.rows.items():
                if name != "primary" and not observed[name] and held and not occupied.intersection(held):
                    fresh.append(name)
                    occupied.update(held)
            if not fresh:
                return None
            found = shortlist(primary, metric, allow_votes, (set(primary) & set(caps)) - set(blocked))
            if found is None:
                return None
            anchor, challenger, groups, paired = found
            members = sorted(set(anchor) | set(challenger), key=lambda n:(-primary[n]["score"], identity(n)))
            per_fold = sum(caps[n]["classifier_fits"] for n in members)
            reserve = max(sum(caps[n]["refit_classifier_fits"] for n in g) for g in groups)
            if remaining > per_fold * len(fresh) + reserve + 1 and menu[0]["action"] == "explore":
                return None
            fresh = fresh[:max(0, (remaining-reserve)//per_fold)]
            if not fresh:
                return None
            self.plan = dict(anchor=list(anchor), challenger=list(challenger), groups=[list(g) for g in groups],
                             members=members, fresh=fresh, reserve=reserve, primary_comparison=paired,
                             remaining_at_lock=remaining, legal_measured_singles=len(primary))
            self.logger.write_json("finalist_lock.json", self.plan)
        plan = self.plan
        if any(n in blocked or n not in caps for n in plan["members"]):
            self.cancelled = True
            self.logger.write_json("finalist_result.json", dict(status="unavailable", members=plan["members"]))
            return None
        missing = [(f,n) for f in plan["fresh"] for n in plan["members"] if n not in observed[f]]
        total = sum(caps[n]["classifier_fits"] for _,n in missing) + plan["reserve"]
        if total > remaining:
            self.cancelled = True
            self.logger.write_json("finalist_result.json", dict(status="insufficient_remaining_fits", required=total))
            return None
        if missing:
            fold, name = missing[0]
            return dict(action="confirm", members=[name], split=fold, utility=primary[name]["score"],
                        reason="Measure the locked alternatives on the same previously unused training groups.",
                        evidence=dict(reserved_refits=total-caps[name]["classifier_fits"],
                                      target_members=plan["challenger"], anchor=plan["anchor"], locked_folds=plan["fresh"]))
        return self.finish(observed, metric, caps)

    def finish(self, observed, metric, caps):
        plan = self.plan
        folds = ["primary", *plan["fresh"]]
        groups = [tuple(g) for g in plan["groups"]]
        scores = {g: {f: observation(observed[f], g, metric)["score"] for f in folds} for g in groups}
        def mean(group, which):
            return float(np.average([scores[group][f] for f in which], weights=[len(self.rows[f]) for f in which]))
        anchor = tuple(plan["anchor"])
        singles = [g for g in groups if len(g)==1 and mean(g,plan["fresh"])>=mean(anchor,plan["fresh"])]
        best_single = min(singles,key=lambda g:(-mean(g,folds), identity(g)))
        eligible = [best_single]
        for group in groups:
            if len(group) < 2:
                continue
            checks = {*subsets(group), best_single, anchor}
            pooled_advantage = min(mean(group,folds)-mean(base,folds) for base in checks)
            corroborated = all(scores[group][f]>=scores[best_single][f] for f in plan["fresh"])
            if (pooled_advantage >= .003 and corroborated and
                    all(mean(group,plan["fresh"])>mean(base,plan["fresh"]) for base in checks)):
                eligible.append(group)
        winner = min(eligible, key=lambda g:(-mean(g,folds), len(g), identity(g)))
        result = dict(status="completed", selected=list(winner), anchor=list(anchor), folds=folds,
                      comparisons=[dict(members=list(g), scores=scores[g], pooled=mean(g,folds)) for g in groups])
        self.logger.write_json("finalist_result.json",result)
        return dict(action="stop", members=list(winner), split=None, utility=mean(winner,folds),
                    reason="Select the deployable model using the same held-out training groups for every finalist.",
                    evidence=dict(scores=scores[winner], refit_fits=sum(caps[n]["refit_classifier_fits"] for n in winner),
                                  matched_finalists=result))
