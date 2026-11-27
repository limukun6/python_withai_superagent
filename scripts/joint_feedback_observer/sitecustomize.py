"""Keep answer files out of every agent process and prediction packages out of fitting subprocesses."""
import os
from pathlib import Path
import runpy
import sys

if os.environ.get('AB_B1_OFFLINE') == '1':
    denied = [Path(os.environ[key]).resolve() for key in ('JOINT_OUTER_ANSWERS', 'JOINT_UPSTREAM_SOURCES')]
    query = Path(os.environ['JOINT_PREDICTION_INPUT']).resolve()
    worker = 'superagent.experiment_worker' in sys.orig_argv

    def isolate(event, args):
        if event == 'open' and isinstance(args[0], (str, bytes)):
            name = os.path.realpath(os.fsdecode(args[0]))
            if any(name == str(p) or name.startswith(str(p) + os.sep) for p in denied) or (worker and name == str(query)):
                raise PermissionError('Held-out data are sealed from this role')

    sys.addaudithook(isolate)
    runpy.run_path(str(Path(__file__).resolve().parents[1] / 'reflection_study_observer/sitecustomize.py'))
