"""Offline A1-01 probes. No API, no completed classifier fits; unique output required."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=('baseline', 'fixed'), required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    def write(name, value):
        (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    write('protocol.json', dict(phase=args.phase, utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p.relative_to(REPO)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((REPO/'superagent').glob('*.py'))},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        python=platform.python_version(), dependencies={n:importlib.metadata.version(n) for n in ('numpy','scipy','scikit-learn')},
        scope='Training-input interception and contract probes, not performance evidence',api_requests=0))
    rows = []
    def record(name, observed, passed):
        rows.append(dict(case=name, observed=observed, passes_fixed_requirement=bool(passed)))
        write('results.json', rows)
    try:
        import numpy as np
        from superagent.models import search_candidates
        from superagent.recipe_models import RecipeTextModel
        from superagent.task_semantics import validate_task_semantics, swap_behavior
        task = lambda meaning: SimpleNamespace(kind='pair_text', spec={'input_schema':{'pair_semantics':meaning}})
        recipes = [c for c in search_candidates('pair_text','semantic') if c.family.startswith('recipe_')]
        report = validate_task_semantics(task('symmetric'), 'semantic')
        for c in recipes:
            captured = {}
            class BoundaryReached(Exception): pass
            labels = np.asarray([0,1,1,0])
            def intercept(self, x, y):
                captured.update(rows=len(y), flipped_second_half=bool(len(y)==8 and np.array_equal(y[4:],1-labels)))
                raise BoundaryReached()
            try:
                with patch('superagent.recipe_models.LogisticRegression.fit',intercept):
                    RecipeTextModel(c.params).fit([('red apple','green pear'),('blue sky','rain cloud'),
                        ('fast car','slow bus'),('big dog','small cat')], labels)
            except BoundaryReached: pass
            excluded = c.family in report['excluded_candidates']
            record(c.family, dict(captured=captured,excluded=excluded), bool(captured.get('flipped_second_half') and excluded))
        record('symmetric_catalog',report,report['status']=='rejected' and report['remaining_candidates']==0)
        pref = validate_task_semantics(task('preference'),'semantic')
        record('preference_compatible',pref,pref['status']=='accepted' and not pref['excluded_candidates'])
        for value in (.5,.52):
            probe=swap_behavior([value,value],[value,value])
            record('ambiguous_'+str(value),probe,probe['pattern']=='ambiguous')
        try:
            swap_behavior([1.1],[.2])
            rejected=False
        except ValueError: rejected=True
        record('invalid_probability',dict(rejected=rejected),rejected)
        write('completion.json', dict(status='complete',phase=args.phase,cases=len(rows),
            passed=sum(r['passes_fixed_requirement'] for r in rows),
            elapsed_seconds=time.monotonic()-start,api_requests=0,classifier_fits=0,
            training_boundary_interceptions=len(recipes)))
        print(json.dumps(dict(phase=args.phase,passed=sum(r['passes_fixed_requirement'] for r in rows),cases=len(rows))))
        return int(args.phase=='fixed' and not all(r['passes_fixed_requirement'] for r in rows))
    except Exception as error:
        write('completion.json',dict(status='failed',error=f'{type(error).__name__}: {error}',
            api_requests=0,classifier_fits=0,elapsed_seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
