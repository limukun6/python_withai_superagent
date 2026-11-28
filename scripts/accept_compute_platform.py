"""Exercise the regular solver and its frozen prediction on a clean CPU-only installation."""
import csv
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from sklearn.datasets import load_wine
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split

from scripts.run_planner_self_review import packet, source_identity
from scripts.unified_released_study import ROOT, read, require, sha


def check():
    missing = [name for name in ('torch', 'transformers', 'catboost') if find_spec(name) is None]
    require(len(missing) == 3, 'This acceptance case needs a clean CPU-only Python environment')
    x,y = load_wine(return_X_y=True)
    x_train,x_query,y_train,y_held = train_test_split(x, y, test_size=.25, stratify=y, random_state=822)
    with tempfile.TemporaryDirectory() as temp:
        folder = Path(temp)
        weights = folder / 'weights'
        weights.mkdir()
        observation, queries = folder / 'training.zip', folder / 'queries.zip'
        packet(observation, 'tabular', 'balanced_accuracy', x_train, y_train)
        packet(queries, 'tabular', 'balanced_accuracy', x_query)
        env = dict(os.environ, LOKY_MAX_CPU_COUNT='2', SUPERAGENT_OFFLINE='1', HF_HUB_OFFLINE='1',
                   OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        for name in list(env):
            if name.startswith('SUPERAGENT_') and name != 'SUPERAGENT_OFFLINE':
                del env[name]
        run = folder / 'run'
        for phase in ('solve', 'replay'):
            args = (['solve', observation, '--prediction-input', queries, '--time-limit', '180']
                    if phase == 'solve' else ['predict', queries])
            subprocess.run([sys.executable, '-m', 'unified_agent', *args, '--weights-dir', weights,
                            '--run-dir', run, '--output', folder / (phase + '.csv')],
                           cwd=ROOT, env=env, check=True, stdin=subprocess.DEVNULL, timeout=260)
        output = folder / 'replay.csv'
        outcome = read(run / 'agent/run_outcome.json')
        machine = read(run / 'agent/compute_environment.json')
        limit = read(run / 'training_protocol.json')['fit_limit']
        require(not machine['accelerators'] and 0 < machine['cpu_threads'] <= 2 and limit == 12,
                'Resource admission ignored the CPU-only receiver')
        require(outcome['status'] == 'completed' and 0 < outcome['attempted_classifier_fits'] <= limit
                and outcome['new_api_requests'] == 0 and sha(output) == sha(folder / 'solve.csv'),
                'Incomplete or inconsistent independent delivery')
        with output.open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        require([r['row_id'] for r in rows] == list(map(str, range(len(y_held)))), 'Missing or reordered rows')
        labels = [int(r['prediction']) for r in rows]
        return dict(system=machine['system'], architecture=machine['architecture'], cpu_threads=machine['cpu_threads'],
                    absent=missing, fit_ceiling=limit, classifier_fits=outcome['attempted_classifier_fits'],
                    members=outcome['members'], correct=sum(int(a==b) for a,b in zip(labels,y_held)), rows=len(rows),
                    held_out_balanced_accuracy=float(balanced_accuracy_score(y_held,labels)), replay=sha(output),
                    solver=source_identity(ROOT), model_api=0)


if __name__ == '__main__':
    result = check()
    print(json.dumps(result), flush=True)
    if os.getenv('GITHUB_STEP_SUMMARY'):
        with Path(os.environ['GITHUB_STEP_SUMMARY']).open('a',encoding='utf-8') as summary:
            summary.write('\n```json\n' + json.dumps(result, indent=2) + '\n```\n')
