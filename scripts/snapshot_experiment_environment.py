"""Portable direct dependency pins and non-secret runtime identity, not pip URLs or credentials."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys

from scripts.a_batch_common import save
from superagent.ab_run import sources


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);root=Path(__file__).resolve().parents[1];pins={}
    for line in (root/'requirements.txt').read_text(encoding='utf-8').splitlines():
        if not line.strip() or line.lstrip().startswith('#'):continue
        match=re.match(r'^([A-Za-z0-9_.-]+)(?:[<>=!~].*)?$',line.strip())
        if not match:raise ValueError('Unexpected requirement syntax; do not export URLs')
        name=match.group(1);pins[name]=importlib.metadata.version(name)
    (args.output/'requirements.direct.lock.txt').write_text(''.join(f'{name}=={version}\n' for name,version in pins.items()),encoding='utf-8')
    save(args.output/'runtime.json',dict(python=platform.python_version(),implementation=platform.python_implementation(),
        operating_system=platform.platform(),machine=platform.machine(),processor=platform.processor(),logical_CPUs=os.cpu_count(),
        dependencies=pins,source_files=sources(),configured_training_threads=2,training_device='CPU',
        fresh_dependency_install_tested=False,scope='Actual installed versions during A44/A47/A50 source epoch; no environment mutation'))
    print(json.dumps(dict(dependencies=pins,python=platform.python_version(),configured_training_device='CPU')))
