"""Prepare B's two six-arm blocks and verify frozen inputs, without model training or API calls."""
import argparse
import gzip
import os
from pathlib import Path

from scripts import run_b1_ab_acceptance as fixture, run_b1_05_pair_pilot as pilot
from scripts import run_bq_01_portfolios as portfolios
from superagent.ab_run import sources, validate_options
from superagent.ab_sentence import verified_directory
from superagent.cli import build_parser


TASKS = ('headlines', 'bean')
ARMS = {'L0': ('local','fixed'), 'L1': ('local','adaptive'),
        'R0': ('raw-history','fixed'), 'R1': ('raw-history','adaptive'),
        'E0': ('checked-history','fixed'), 'E1': ('checked-history','adaptive')}
ORDER_SALT = 'AIAA3102-C1-B-order-20261121'


def option_probes(task_file):
    out = {}
    for method in ('local','raw-history','checked-history'):
        opts = build_parser().parse_args(['run',str(task_file),'--scheduler','ab-v1','--portfolio','ab-v1',
                                          '--planner',method,'--fit-limit','11'])
        try:
            limit = validate_options(opts)
            out[method] = dict(validated=True, physical_fit_limit=limit)
        except ValueError as error:
            out[method] = dict(validated=False, reason=str(error))
    return out


def prepare(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    fixture.offline(root)
    os.environ.update(SUPERAGENT_OFFLINE='1', SUPERAGENT_MINILM_WEIGHTS=str(args.weights.resolve()))
    source = fixture.digest(sources())
    records, blocks = {}, {}
    fixture.write(root / 'protocol.json', dict(batch='BC1-01-prepare', production_source_sha256=source,
        runner_sha256=fixture.sha(Path(__file__).read_bytes()),
        plan_sha256=fixture.sha((fixture.ROOT/'docs/batches/BC1_01_preparation.md').read_bytes()),
        environment=pilot.environment(), order_salt=ORDER_SALT, outer_seed=pilot.OUTER_SEED, inner_seed=pilot.INNER_SEED,
        groups=ARMS, resources_per_arm=dict(fit_limit=11, reserved_refit=1, wall_seconds=pilot.WALL, trial_seconds=pilot.TRIAL),
        execution_budget_this_preparation=dict(classifier_fits=0, model_api_requests=0)))
    try:
        if verified_directory() is None:
            raise ValueError('Reference portfolio requires the verified frozen MiniLM')
        reference_file = args.q_reference / 'protocol.json'
        reference = fixture.read(reference_file)
        for task in TASKS:
            expected = reference['tasks'][task + '_expanded']
            actual = portfolios.prepare_shared(root, task, args)
            checks = {k: actual[k] == expected[k] for k in ('source','semantics','files','splits','row_ids_sha256','candidate_ids','catalog_sha256')}
            for label, file in [('reference',args.q_reference/task/'row_ids.json.gz'),('prepared',root/task/'row_ids.json.gz')]:
                checks[label+'_row_file'] = fixture.sha(gzip.decompress(file.read_bytes())) == expected['row_ids_sha256']
            if not all(checks.values()):
                raise ValueError('Reference differs on ' + task + ': ' + ', '.join(k for k,v in checks.items() if not v))
            records[task] = dict(checks=checks, source=actual['source'], files=actual['files'], splits=actual['splits'],
                row_ids_sha256=actual['row_ids_sha256'], catalog_sha256=actual['catalog_sha256'],
                candidate_ids=actual['candidate_ids'], training_rows=actual['inner_input_rows'], outer_rows=actual['outer_rows'])
            order = sorted(ARMS, key=lambda arm: fixture.sha((ORDER_SALT+'/'+task+'/'+arm).encode()))
            blocks[task] = [dict(arm=arm, planner=ARMS[arm][0], allocation=ARMS[arm][1],
                request_limit=0 if ARMS[arm][0]=='local' else 3, anchor=actual['candidate_ids'][0]) for arm in order]
        fixture.write(root/'results.json', dict(tasks=records, scheduled_blocks=blocks, q_input_protocol_sha256=fixture.sha(reference_file.read_bytes()),
            base_AB_option_probes=option_probes(root/'headlines'/'train.zip'),
            q_source_matches=reference['source_sha256'] == source, q_environment_matches=reference['environment'] == pilot.environment()))
        fixture.write(root/'completion.json', dict(status='inputs_prepared', planned_paths=12, executed_paths=0,
            planned_max_fits=132, planned_max_requests=24, actual_fits=0, actual_api_requests=0, outer_scored=False))
    except Exception as error:
        fixture.write(root/'completion.json', dict(status='failed', error=str(error), actual_fits=0, actual_api_requests=0))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('q-reference','headlines','uci','weights','output'):
        p.add_argument('--'+name, type=Path, required=True)
    prepare(p.parse_args())
