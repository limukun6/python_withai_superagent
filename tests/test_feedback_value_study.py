from unittest import TestCase

from superagent.benchmark import Policy
from superagent.confirmation import shortlist
from superagent.contracts import DecisionState, ExistingPolicyAdapter
from superagent.evidence_schedule import catalog_for
from tools.feedback_value_study import MaskedSearch


class FeedbackValueStudyTests(TestCase):
    def test_masking_changes_only_what_search_can_read(self):
        catalog = catalog_for("categorical_tabular")
        configurations = [{k: row[k] for k in ("id", "params")} for row in catalog]
        def baseline():
            return ExistingPolicyAdapter(Policy("local_interleaved_routes", 707, None, None, 3, configurations))

        def state(history):
            used = {row["candidate"] for row in history}
            available = [row for row in configurations if row["id"] not in used]
            return DecisionState.build({"primary_metric": "accuracy"}, history, available, 4)

        def trial(index, config, score):
            return dict(trial=index, candidate=config["id"], params=config["params"],
                        status="completed", primary_metric="accuracy", search_score=score,
                        search_accuracy=score)

        first = trial(1, configurations[0], 0.1)
        second = baseline().propose(state([first]))
        next_config = {"id": second.candidate, "params": second.params}
        observed = [first, trial(2, next_config, 0.9)]
        swapped = [trial(1, configurations[0], 0.9), trial(2, next_config, 0.1)]
        base_actions, masked_actions = [], []
        for history in (observed, swapped):
            base, masked = baseline(), MaskedSearch(catalog, 707)
            self.assertEqual(base.propose(state(history[:1])).candidate, second.candidate)
            self.assertEqual(masked.propose(state(history[:1])).candidate, second.candidate)
            base_actions.append(base.propose(state(history)).candidate)
            masked_actions.append(masked.propose(state(history)).candidate)
        self.assertNotEqual(*base_actions)
        self.assertEqual(*masked_actions)
        self.assertEqual(shortlist(observed, 1)[0]["candidate"], second.candidate)
        self.assertEqual(shortlist(swapped, 1)[0]["candidate"], first["candidate"])
