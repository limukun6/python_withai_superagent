"""Reproduce the reviewed transplant in a fresh, clean e5ee263 checkout only."""
import argparse
import hashlib
from pathlib import Path
import re
import subprocess
from scripts.a_batch_common import save


def replace_once(text,before,after):
    if text.count(before)!=1:raise ValueError('Reviewed patch context changed')
    return text.replace(before,after,1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkout','patches','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    def git(*args):return subprocess.run(['git',*args],cwd=a.checkout,capture_output=True,text=True,encoding='utf-8')
    if git('status','--porcelain').stdout:raise ValueError('Fresh clean checkout required; do not overwrite work')
    if git('rev-parse','HEAD').stdout.strip()!=git('rev-parse','e5ee263').stdout.strip():raise ValueError('Exact registered A125 base required')
    expected={'A117_B_joint_encoder.patch':'73f5c60192145ada09a162f2786bf0c2bf6d486565f27a065e751988ecb17863',
              'A117_B_single_conflict.patch':'cb73c95ab070388e7c770edfee7e0dfbe383883dd77e98caef79b239181a699b'}
    for name,digest in expected.items():
        if hashlib.sha256((a.patches/name).read_bytes()).hexdigest()!=digest:raise ValueError('B patch bytes changed')
    result=git('apply','--3way',str(a.patches/'A117_B_joint_encoder.patch'))
    (a.output/'joint_apply.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    # The only expected three-way conflict is A120's explicit diagnostics guard in CLI.
    conflicts=git('diff','--name-only','--diff-filter=U').stdout.splitlines()
    if conflicts!=['superagent/cli.py']:raise ValueError('Unexpected transplant conflict set: '+repr(conflicts))
    path=a.checkout/'superagent/cli.py';text=path.read_text(encoding='utf-8')
    match=re.search(r'<<<<<<< ours\n(.*?)=======\n(.*?)>>>>>>> theirs\n',text,re.S)
    if not match or 'parameter_semantics' not in match[1] or 'cover_untried_encoder' not in match[2]:raise ValueError('Unexpected CLI conflict')
    guard="    if getattr(args,'parameter_semantics','off')!='off' and args.scheduler not in ('ab-v1','ab-fixed','ab-finalists','ab-complement','ab-adaptive-complement'):\n        raise ValueError('Parameter semantics diagnostics require the shared A/B executor')\n"
    text=text[:match.start()]+guard+match[2]+text[match.end():];path.write_text(text,encoding='utf-8',newline='\n')
    if git('add','superagent/cli.py').returncode:raise RuntimeError('Unable to resolve reviewed CLI conflict')
    result=git('apply','--exclude=superagent/cli.py',str(a.patches/'A117_B_single_conflict.patch'))
    (a.output/'conflict_apply.log').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:raise RuntimeError('Single-conflict module patch failed')
    text=path.read_text(encoding='utf-8')
    anchor="    catalog = getattr(args, 'preference_catalog', 'standard')\n"
    text=replace_once(text,anchor,anchor+"    if getattr(args, 'resolve_single_conflict', False) and args.scheduler != 'ab-adaptive-complement':\n        raise ValueError('Single-model conflict resolution requires the adaptive complement scheduler')\n")
    anchor="    run.add_argument('--catalog-scope',"
    text=replace_once(text,anchor,"    run.add_argument('--resolve-single-conflict', action='store_true',\n        help='Compare disagreeing single-model leaders on a third split when the remaining budget permits')\n"+anchor)
    path.write_text(text,encoding='utf-8',newline='\n')
    path=a.checkout/'tools/ab_registered_run.py';text=path.read_text(encoding='utf-8')
    text=replace_once(text,"args.scheduler not in ('ab-v1','ab-fixed')","args.scheduler not in ('ab-v1','ab-fixed','ab-finalists','ab-complement','ab-adaptive-complement')")
    text=replace_once(text,'def from_rows(task, directory, seed, metric):','def from_rows(task, directory, seed, metric, **options):')
    text=replace_once(text,'return freeze(task,directory,seed,metric)','return freeze(task,directory,seed,metric,**options)')
    path.write_text(text,encoding='utf-8',newline='\n')
    save(a.output/'completion.json',dict(status='transplanted_pending_tests',actual_classifier_fits=0,real_API_requests=0))


if __name__=='__main__':main()
