"""Use a verified local encoder environment while forwarding the single official CLI."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    assets=a.assets.resolve();completion=json.loads((assets/'completion.json').read_text(encoding='utf-8'))
    environment=json.loads((assets/'environment.json').read_text(encoding='utf-8'))
    if completion['status']!='ready':raise ValueError('Encoder assets are not ready')
    python=Path(environment['executable'])
    if not python.is_file():raise ValueError('Registered runtime interpreter is unavailable; prepare assets on this machine')
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:p.error('Pass the original superagent command after --')
    env={**os.environ,'SUPERAGENT_MINILM_WEIGHTS':str(assets/'minilm'),
         'SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS':str(assets/'mobilenet_v3_small-047dcff4.pth'),
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    if a.dry_run:
        print(json.dumps(dict(runtime=str(python),forwarded_command=command,assets=str(assets),
            executed=False,actual_classifier_fits=0,real_api_requests=0),ensure_ascii=False));return 0
    return subprocess.run([str(python),'-m','superagent',*command],cwd=Path(__file__).resolve().parents[1],env=env).returncode


if __name__=='__main__':raise SystemExit(main())
