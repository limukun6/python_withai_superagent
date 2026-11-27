"""Spend surplus fits on unexplored representations while retaining a confirmed decision."""
import numpy as np

from .experiment_policy import identity, measured_options


class CoverageReview:
    def __init__(self):
        self.incumbent = None
        self.pending = None
        self.probes = []
        self.events = []

    def record(self):
        return dict(probes=list(self.probes), incumbent=self.incumbent, pending=self.pending, events=list(self.events))

    def choose(self, choice, menu, caps, observed, metric, folds, remaining, blocked, seconds_left, history,
               *, allow_votes=True):
        if self.incumbent is None:
            if choice["action"] != "stop":
                return choice
            scores = choice["evidence"]["scores"]
            if not all(fold in scores for fold in folds) or len(folds) < 2 or min(scores.values()) >= 1:
                return choice
            self.incumbent = choice
            self.events.append(dict(event="protected", members=choice["members"], scores=scores))

        incumbent = self.incumbent
        base = tuple(sorted(incumbent["members"]))
        options = measured_options(observed, metric, allow_votes=allow_votes)
        primary = folds[0]
        base_cost = sum(caps[m]["refit_classifier_fits"] for m in base)
        elapsed = [row["training_seconds"] for row in history if row["status"] == "completed"]
        if seconds_left <= (base_cost + 3) * max(elapsed, default=1.) + 15:
            return incumbent

        if self.pending:
            name = self.pending["probe"]
            if name in blocked or name not in observed[primary]:
                self.events.append(dict(event="probe_failed", probe=name))
                self.pending = None
            else:
                group = self.pending.get("nominee")
                if group is None:
                    eligible = []
                    for members, values in options.items():
                        if name not in members or values[primary] <= options[base][primary]:
                            continue
                        missing = [(m, f) for f in folds[1:] for m in members if m not in observed[f]]
                        reserve = max(base_cost, sum(caps[m]["refit_classifier_fits"] for m in members))
                        if any(m in blocked for m, _ in missing) or sum(caps[m]["classifier_fits"] for m, _ in missing) + reserve > remaining:
                            continue
                        eligible.append((members, values[primary]))
                    if eligible:
                        group, score = min(eligible,key=lambda item:(-item[1], len(item[0]), identity(item[0])))
                        self.pending["nominee"] = list(group)
                        self.events.append(dict(event="nomination_locked", probe=name, members=list(group),
                                                primary=score, incumbent=options[base][primary]))
                    else:
                        self.events.append(dict(event="no_affordable_primary_gain", probe=name))
                        self.pending = None
                if group is not None:
                    members = tuple(group)
                    reserve = max(base_cost, sum(caps[m]["refit_classifier_fits"] for m in members))
                    missing = [(m, f) for f in folds[1:] for m in members if m not in observed[f]]
                    if missing and any(m in blocked for m, _ in missing):
                        self.events.append(dict(event="confirmation_failed", members=list(members)))
                        self.pending = None
                    elif missing:
                        m, f = missing[0]
                        if sum(caps[c]["classifier_fits"] for c, _ in missing) + reserve > remaining:
                            return incumbent
                        return dict(action="confirm", members=[m], split=f, utility=options[members][primary],
                                    reason="Confirm the locked nomination on matched training groups.",
                                    evidence=dict(target_members=list(members),reserved_refits=reserve))
                    else:
                        scores = options[members]
                        changes = [scores[f] - options[base][f] for f in folds[1:]]
                        count = [len(observed[f][members[0]]["labels"]) for f in folds]
                        accepted = min(changes) >= 0 and np.average(changes, weights=count[1:]) > 0
                        self.events.append(dict(event="confirmed" if accepted else "incumbent_retained",
                                                members=list(members), scores=scores, base_scores=options[base]))
                        if accepted:
                            incumbent = dict(action="stop",members=list(members),split=None,
                                utility=float(np.average([scores[f] for f in folds],weights=count)),
                                reason="Locked alternative improved on matching confirmation groups.",
                                evidence=dict(scores=scores,refit_fits=sum(caps[m]["refit_classifier_fits"] for m in members)))
                            self.incumbent = incumbent
                            base_cost = incumbent["evidence"]["refit_fits"]
                        self.pending = None

        # Budget a full set of confirmations and the final refit before opening a new route.
        if len(self.probes) >= 2:
            return incumbent
        viable = []
        for action in menu:
            if action["action"] != "explore":
                continue
            name = action["members"][0]
            evidence = action["evidence"]
            reserve = max(base_cost, caps[name]["refit_classifier_fits"] + (1 if allow_votes else 0))
            if name in self.probes or evidence["novelty"] < .4 or len(folds)*caps[name]["classifier_fits"] + reserve > remaining:
                continue
            viable.append(action)
        if not viable:
            return incumbent
        probe = min(viable, key=lambda a:(-a["evidence"]["novelty"], -a["utility"], identity(a["members"])))
        name = probe["members"][0]
        self.pending = dict(probe=name)
        self.probes.append(name)
        self.events.append(dict(event="new_representation", probe=name, novelty=probe["evidence"]["novelty"],
                                remaining_fits=remaining))
        return dict(probe, reason="Spend surplus evidence budget on a distinct untried representation.",
                    evidence=dict(probe["evidence"], reserved_refits=max(base_cost, caps[name]["refit_classifier_fits"])))
