"""Run the preregistered v2 table projection on the same explicitly seen engineering histories."""
from pathlib import Path
from unittest.mock import patch

from scripts import run_a139_compact_feedback as original

PLAN=original.ROOT/'docs/batches/A139B_table_projection_protocol_20261124_zh.md'


if __name__=='__main__':
    # Preserve the v1 runner and receipts. Only the recorded runner/plan identities
    # change; inputs, gates, tests and fixed state positions stay the same.
    save=original.save
    def journal(path,value):
        if Path(path).name=='protocol.json':
            value={**value,'batch':'A139B','plan_sha256':original.sha(PLAN),
                   'runner_sha256':original.sha(Path(__file__)),
                   'implementation_runner_sha256':original.sha(Path(original.__file__)),
                   'seen_engineering_inputs':True,'prior_variant_commit':'f61cfc3'}
        return save(path,value)
    with patch.object(original,'save',journal):original.main()
