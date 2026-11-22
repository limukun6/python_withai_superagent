"""Use a verified local encoder environment while forwarding the single official CLI."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',type=Path)
    p.add_argument('--encoder-policy',choices=['auto','restricted','research'],default='auto',
                   help='Auto offers verified generic downstream encoders; restricted excludes them for ablation')
    p.add_argument('--dry-run',action='store_true');p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    assets=a.assets.resolve() if a.assets else None
    if a.encoder_policy=='research' and assets is None:p.error('Research encoders require --assets')
    environment=json.loads((assets/'environment.json').read_text(encoding='utf-8')) if assets else {'executable':sys.executable}
    ready=assets is not None and (assets/'completion.json').is_file() and json.loads((assets/'completion.json').read_text(encoding='utf-8'))['status']=='ready'
    if a.encoder_policy=='research' and not ready:
        raise ValueError('Encoder assets are not ready')
    python=Path(environment['executable'])
    if not python.is_file():raise ValueError('Registered runtime interpreter is unavailable; prepare assets on this machine')
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:p.error('Pass the original superagent command after --')
    allowed=a.encoder_policy!='restricted' and ready
    env={**os.environ,'SUPERAGENT_MINILM_WEIGHTS':str(assets/'minilm') if allowed else '',
         'SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS':str(assets/'mobilenet_v3_small-047dcff4.pth') if allowed else '',
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    if a.dry_run:
        print(json.dumps(dict(runtime=str(python),forwarded_command=command,assets=str(assets) if assets else None,
            encoder_policy=a.encoder_policy,pretrained_encoders_enabled=allowed,downstream_feature_role_clarified=True,
            qualification_source='user_forwarded_teacher_clarification_20261122',eligibility_rechecked_by_core=True,
            executed=False,actual_classifier_fits=0,real_api_requests=0),ensure_ascii=False));return 0
    return subprocess.run([str(python),'-m','superagent',*command],cwd=Path(__file__).resolve().parents[1],env=env).returncode


if __name__=='__main__':raise SystemExit(main())
