"""Reproducibly stage B native heads, retaining A controller and explicit per-run catalogue scope."""
import argparse
from pathlib import Path
import shutil
import subprocess
import time
from scripts.a_batch_common import hashes, save
from scripts.run_a10_real_task_pilot import sha

ROOT = Path(__file__).resolve().parents[1]
FILES = ['ab_sentence.py', 'ab_methods.py', 'ab_catalog.py', 'external_assets.py', 'ab_edit_views.py']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('provider', 'target', 'output'):
        p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    before = hashes(a.target)
    provider = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=a.provider, text=True).strip()
    save(a.output/'protocol.json', dict(batch='A112-stage', provider_commit=provider,
         target_before=before, runner_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT/'docs/batches/A112_A114_native_and_H1_queue_plan.md'),
         native_files=FILES, actual_classifier_fits=0, real_API_requests=0))
    try:
        if provider != 'c6584776be99a1eaee59b881047566219651edbd':
            raise ValueError('Unexpected provider')
        if (a.target/'superagent/ab_edit_views.py').exists():
            raise ValueError('Target already modified; use a clean A checkout')
        for name in FILES:
            shutil.copyfile(a.provider/'superagent'/name, a.target/'superagent'/name)
        cli = a.target/'superagent/cli.py'
        value = cli.read_text(encoding='utf-8')
        anchor = "    if args.scheduler in ('ab-v1','ab-fixed','ab-finalists'):\n        from .ab_run import command\n        return command(args)"
        replacement = '''    catalog = getattr(args, 'preference_catalog', 'standard')
    if args.scheduler in ('ab-v1','ab-fixed','ab-finalists'):
        from .ab_methods import PREFERENCE_CATALOG_ENV
        from .ab_run import command
        previous = os.environ.get(PREFERENCE_CATALOG_ENV)
        os.environ[PREFERENCE_CATALOG_ENV] = catalog
        try:
            return command(args)
        finally:
            if previous is None:
                os.environ.pop(PREFERENCE_CATALOG_ENV, None)
            else:
                os.environ[PREFERENCE_CATALOG_ENV] = previous
    if catalog != 'standard':
        raise ValueError('Expanded preference catalog requires the shared A/B executor')'''
        if value.count(anchor) != 1:
            raise ValueError('CLI dispatch anchor changed')
        value = value.replace(anchor, replacement)
        anchor = "    run.add_argument('--catalog-scope',"
        if value.count(anchor) != 1:
            raise ValueError('CLI parser anchor changed')
        value = value.replace(anchor, "    run.add_argument('--preference-catalog', choices=['standard','deep','deep-edit'], default='standard',\n                     help='Optional verified native twelve-layer preference candidates in the shared A/B executor')\n"+anchor)
        cli.write_bytes(value.encode('utf-8'))
        run = a.target/'superagent/ab_run.py'
        value = run.read_text(encoding='utf-8')
        anchor = '            scheduler=args.scheduler, catalog_scope=args.catalog_scope,\n'
        if value.count(anchor) != 1:
            raise ValueError('Run manifest anchor changed')
        value = value.replace(anchor, anchor+"            preference_catalog=getattr(args,'preference_catalog','standard'),\n")
        run.write_bytes(value.encode('utf-8'))
        if sha(a.target/'superagent/prose_claims.py') != sha(ROOT/'superagent/prose_claims.py'):
            raise ValueError('A prose fixes lost')
        target = a.target/'docs/batches/A112_A114_native_and_H1_queue_plan.md'
        shutil.copyfile(ROOT/'docs/batches/A112_A114_native_and_H1_queue_plan.md', target)
        save(a.output/'results.json', dict(target_after=hashes(a.target), native_provider_files={n:sha(a.provider/'superagent'/n) for n in FILES},
             component_controller_integrated=False, A_prose_fixes_preserved=True))
        save(a.output/'completion.json', dict(status='staged_awaiting_tests', actual_classifier_fits=0,
             real_API_requests=0, elapsed_seconds=time.monotonic()-start))
    except Exception as error:
        save(a.output/'completion.json', dict(status='stopped', error=str(error), actual_classifier_fits=0, real_API_requests=0))
        raise


if __name__ == '__main__':
    main()
