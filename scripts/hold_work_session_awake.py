"""Temporary power request while actual report/publication work continues, not an experiment."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import time

from scripts.a_batch_common import save
from tools.unattended_runtime import prevent_idle_sleep


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('window-config','done-file','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    deadline=datetime.fromisoformat(json.loads(args.window_config.read_text(encoding='utf-8-sig'))['deadline'])
    save(args.output/'protocol.json',dict(purpose='Prevent idle standby only while the active work session is being finalized',
        deadline=deadline.isoformat(),permanent_power_policy_changed=False,classifier_fits=0,API_requests=0))
    with prevent_idle_sleep():
        while datetime.now(timezone.utc)<deadline and not args.done_file.exists():time.sleep(5)
    save(args.output/'completion.json',dict(status='idle_sleep_request_released',
        reason='finalization_marker' if args.done_file.exists() else 'registered_deadline',permanent_power_policy_changed=False))
