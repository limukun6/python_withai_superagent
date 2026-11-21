"""Compare original, enumerated, and evidence-composed text choices through A's local CLI."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

from superagent import ab_run, ab_text_recipes
from superagent.cli import build_parser
from superagent.scheduler_bridge import fingerprint
from tools.ab_registered_rows import registered_partitioner


POLICIES = ('original','expanded','composed')


def run(policy, command, manifest=None, sha=None):
    args = build_parser().parse_args(command)
    if (policy not in POLICIES or not command or command[0]!='run' or args.scheduler!='ab-v1'
            or args.planner!='local' or args.portfolio!='ab-v1' or args.catalog_scope!='full'):
        raise ValueError('The B2 comparison requires the complete local AB-v1 research entry')
    if (manifest is None)!=(sha is None):
        raise ValueError('A registered split needs both a file and its hash')
    register,freeze,pick = ab_run.registered_capabilities,ab_run.freeze_partitions,ab_run.next_action
    receipt_dir = None

    def catalogue(task, train_rows, scope='full'):
        menu = register(task,train_rows,scope)
        if policy=='original':
            return {name:cap for name,cap in menu.items() if name not in ab_text_recipes.ROUTES}
        return menu

    def from_rows(task,directory,seed,metric):
        nonlocal receipt_dir
        receipt_dir = directory/'bounded_recipes'
        receipt_dir.mkdir()
        (receipt_dir/'policy.json').write_text(json.dumps({'policy':policy,'input_kind':task.kind},indent=2)+'\n')
        if manifest is None:
            return freeze(task,directory,seed,metric)
        with patch.object(ab_run,'grouped_partition',registered_partitioner(task,manifest,sha)):
            return freeze(task,directory,seed,metric)

    def decide(state,**kwargs):
        evidence = []
        def compose(eligible,verified,caps):
            proof = ab_text_recipes.propose(eligible,verified,caps)
            if proof is not None:
                evidence.append(proof)
            return proof
        if policy=='composed':
            kwargs['on_local_recipe'] = compose
        action = pick(state,**kwargs)
        step = len(state['history'])+1
        record = dict(step=step,state_sha256=fingerprint(state),policy=policy,action=action,proposals=evidence)
        with (receipt_dir/f'step_{step}.json').open('x',encoding='utf-8') as output:
            json.dump(record,output,indent=2,allow_nan=False)
            output.write('\n')
        return action

    with patch.object(ab_run,'registered_capabilities',catalogue), patch.object(ab_run,'freeze_partitions',from_rows), patch.object(ab_run,'next_action',decide):
        return args.handler(args)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',choices=POLICIES,required=True)
    parser.add_argument('--split-manifest',type=Path)
    parser.add_argument('--split-sha256')
    if '--' not in sys.argv[1:]:
        parser.error('Pass the official run command after --')
    boundary = sys.argv.index('--',1)
    options = parser.parse_args(sys.argv[1:boundary])
    raise SystemExit(run(options.policy,sys.argv[boundary+1:],options.split_manifest,options.split_sha256))
