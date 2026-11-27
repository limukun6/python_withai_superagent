"""Explicit user-accepted unresolved-history snapshot, scoped to one real ledger and context."""
from contextlib import contextmanager
from unittest.mock import patch



from superagent.accepted_history import row_digest,load_snapshot,check_snapshot


@contextmanager
def accepted_history(path,expected_sha256):
    """Keep all unknown costs null; only the pre-existing named rows are admitted.

    The production planner/executor stays byte-identical to frozen L0 models.
    Checks run inside the ledger's normal transactions, including each reserve.
    """
    from superagent import research_budget
    record=load_snapshot(path,expected_sha256);original_annotate=research_budget.annotate
    def check(db,state,amount):return check_snapshot(db,state,amount,record)
    def annotate(state):
        return {**original_annotate(state),'history_policy':'user-accepted-snapshot-research',
            'accepted_history_sha256':expected_sha256,'accepted_unknown_request_count':len(record['accepted_unknown_requests']),
            'policy_note':'User accepts enumerated existing unknown charges without settlement. Costs stay null; any new unknown/pending blocks. Known-cost limit remains enforced.'}
    with patch.object(research_budget,'_check',check),patch.object(research_budget,'annotate',annotate):
        yield record
