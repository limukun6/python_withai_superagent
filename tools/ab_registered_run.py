"""Use a previously signed row manifest with the official A/B run handler."""
import argparse
from pathlib import Path
import sys
from unittest.mock import patch

from superagent import ab_run
from superagent.cli import build_parser
from tools.ab_registered_rows import registered_partitioner


def run(manifest, sha, command):
    args = build_parser().parse_args(command)
    if not command or command[0] != 'run' or args.scheduler not in ('ab-v1','ab-fixed'):
        raise ValueError('An official AB scheduling run is required')
    freeze = ab_run.freeze_partitions

    def from_rows(task, directory, seed, metric):
        selector = registered_partitioner(task, manifest, sha)
        with patch.object(ab_run,'grouped_partition',selector):
            return freeze(task,directory,seed,metric)

    with patch.object(ab_run,'freeze_partitions',from_rows):
        return args.handler(args)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split-manifest',required=True,type=Path)
    parser.add_argument('--split-sha256',required=True)
    if '--' not in sys.argv[1:]:
        parser.error('Pass the official run command after --')
    boundary=sys.argv.index('--',1)
    options=parser.parse_args(sys.argv[1:boundary])
    raise SystemExit(run(options.split_manifest,options.split_sha256,sys.argv[boundary+1:]))
