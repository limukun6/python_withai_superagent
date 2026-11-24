"""Run one bounded command with a temporary idle-sleep guard and no retry."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from scripts.a_batch_common import save
from tools.unattended_runtime import prevent_idle_sleep


def run(args):
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise ValueError('max-seconds must be finite and positive')
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        raise ValueError('Provide an executable and arguments after --')
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    protocol = dict(batch='unattended-command', started_at=datetime.now(timezone.utc).isoformat(),
        maximum_seconds=args.max_seconds, command_sha256=hashlib.sha256(json.dumps(command).encode()).hexdigest(),
        executable=Path(command[0]).name, automatic_retries=False,
        persistent_power_policy_changed=False, display_required=False)
    save(root / 'protocol.json', protocol)
    child = None
    record = dict(status='stopped', child_started=False, automatic_retry=False)
    try:
        with prevent_idle_sleep(), (root / 'child.log').open('w', encoding='utf-8') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            record['child_started'] = True
            try:
                code = child.wait(timeout=args.max_seconds)
            except subprocess.TimeoutExpired:
                subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True, check=False)
                child.wait(timeout=15)
                record.update(error='outer_time_limit', child_exit_code=child.returncode)
                return 124
            record.update(status='completed' if code == 0 else 'stopped', child_exit_code=code)
            return code
    except Exception as error:
        record.update(status='stopped', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        if child is not None and child.poll() is None:
            subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True, check=False)
            child.wait(timeout=15)
        record.update(elapsed_seconds=time.monotonic()-start, finished_at=datetime.now(timezone.utc).isoformat(),
            persistent_power_policy_changed=False)
        save(root / 'completion.json', record)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    raise SystemExit(run(parser.parse_args()))
