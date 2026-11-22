"""Qualify the matrix coordinator using an actual official run without opening evaluation labels."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from pathlib import Path
from threading import Barrier

from scripts import run_b1_ab_acceptance as rec
from scripts import run_b_c1_matrix as matrix


def rejected(fn, message):
    try:
        fn()
    except ValueError as error:
        if message not in str(error):
            raise
        return str(error)
    raise AssertionError('Expected rejection: '+message)


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    rec.offline(root)
    os.environ['SUPERAGENT_MINILM_WEIGHTS'] = str(args.weights.resolve())
    development = rec.read(args.development/'protocol.json')
    protocol = matrix.freeze_inputs(args.portable)
    record = dict(batch='BC1-03',matrix=protocol,development_reference=rec.digest(development),
                  code=matrix.ab_run.sha(Path(__file__)),plan=matrix.ab_run.sha(rec.ROOT/'docs/batches/BC1_03_runner_plan.md'),
                  fits_allowed=0,model_api_limit=0,outer_scores_allowed=0)
    rec.write(root/'protocol.json',record)
    try:
        expected = dict(development['mrpc'],anchor=development['mrpc']['candidate_ids'][0])
        official = args.development/'mrpc/run'
        def verify(task):
            return matrix.verify_run(official,development['source'],task,'L1')
        full = verify(expected)
        tests = {}
        for section, key, message in [('files','train.zip','Run identity'),('splits','primary','partition identity')]:
            changed = deepcopy(expected)
            if section=='splits':
                changed[section][key]['row_ids_sha256']='0'*64
            else:
                changed[section][key]='0'*64
            tests[section] = rejected(lambda:verify(changed),message)
        tests['as_formal_c1_arm'] = rejected(lambda:matrix.verify_run(official,protocol['source'],
                                                              protocol['tasks']['headlines'],'L1'),'Run identity')
        tests['incomplete_matrix'] = rejected(lambda:matrix.frozen_models(root/'incomplete',protocol),'missing')
        # One race for the same arm, with no actual worker or request.
        arm = root/'resumption'/'arm'
        marker = dict(protocol=rec.digest(record),task='receipt_only',arm='L1')
        start = Barrier(2)
        def attempt():
            start.wait()
            try:
                return 'acquired' if matrix.claim_or_resume(arm,marker) else 'resumed'
            except ValueError:
                return 'prior_claim'
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(attempt) for _ in range(2)]
            racing = sorted(f.result() for f in futures)
        if racing != ['acquired','prior_claim']:
            raise ValueError('Two callers could claim the same arm')
        tests['interrupted_before_outcome'] = rejected(lambda:matrix.claim_or_resume(arm,marker),'no final outcome')
        tests['changed_intent'] = rejected(lambda:matrix.claim_or_resume(arm,dict(marker,arm='E0')),'matching launch intent')
        (arm/'official').mkdir()
        (arm/'official/run_outcome.json').write_bytes((official/'run_outcome.json').read_bytes())
        if matrix.claim_or_resume(arm,marker):
            raise ValueError('A completed child could be launched again')
        rec.write(root/'results.json',dict(registered_orders={name:t['order'] for name,t in protocol['tasks'].items()},
                  actual_development_run_revalidated=full,negative_controls=tests,concurrent_claims=racing,
                  observed_finished_child_launches_another=False))
        rec.write(root/'completion.json',dict(status='passed',accepted_cli_option_sets=12,
                  production_preflight_or_api_requests=0,development_model_audits=1,formal_c1_paths_executed=0,
                  fresh_fits=0,outer_scores=0))
    except Exception as error:
        rec.write(root/'completion.json',dict(status='failed',reason=str(error),fresh_fits=0,model_api_requests=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('portable','development','weights','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
