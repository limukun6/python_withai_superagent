"""Import the study branches into a clean A repository and reproduce frozen predictions."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


BASE = 'd910dc4dac3a2cafda654d29d5de8c5a86b0f8e8'
STABLE = 'codex/b-a31-joint'
ROUTE = 'codex/b-route-representative'
KERNEL = 'codex/b-nonlinear-frozen-text'
RECIPE = 'codex/b-evidence-text-recipes'
CANONICAL = 'codex/b-canonical-preference-output'


def call(cmd, cwd, env=None):
    return subprocess.check_output(list(map(str,cmd)),cwd=cwd,env=env,text=True,stderr=subprocess.STDOUT,timeout=240).strip()


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(1024*1024),b''):
            digest.update(part)
    return digest.hexdigest()


def get(obj, keys):
    for key in keys:
        obj = obj[key]
    return obj


def cases(work):
    anon = work/'ab_joint_a31/runs/ba31_anonymous_01/case'
    anon_data = work/'ab_joint_a27/runs/ba27_from_5ed_01/runs/ba27_anon_01/test.zip'
    route = work/'ab_route_representative/runs/route_coupons_01'
    kernel = work/'ab_nonlinear_text/runs'
    sms = work/'ab_refit_time/runs/prospective_clock_01/sms/features.zip'
    titles = work/'ab_contextual_preference/runs/context_headlines_02/45101/features.zip'
    b2 = work/'ab_bounded_recipe/runs/b2_composition_01/sms_composed'
    return [
        (STABLE,'anonymous',anon,anon_data,31,'ba31_b_anonymous',['model_sha256'],['prediction_sha256']),
        (ROUTE,'coupons',route/'45102_representative',route/'45102/features.zip',1627,'b_route_coupons',
            ['paths','45102_representative','model_sha256'],['pairs','45102','grade','representative','csv_sha256']),
        (KERNEL,'headlines',kernel/'text_kernel_titles_01/45101',titles,1672,'b_text_kernel_titles',
            ['new_paths','45101','model_sha256'],['pairs','45101','grades','extended','csv_sha256']),
        (KERNEL,'sms_kernel',kernel/'text_kernel_sms_01/sms',sms,825,'b_text_kernel_sms',
            ['new_paths','sms','model_sha256'],['pairs','sms','grades','extended','csv_sha256']),
        (KERNEL,'film',kernel/'text_kernel_domains_01/imdb_extended',kernel/'text_kernel_domains_01/imdb/features.zip',
            1000,'b_text_kernel_domains',['paths','imdb_extended','model_sha256'],['pairs','imdb','grades','extended','csv_sha256']),
        (RECIPE,'sms_recipe',b2,sms,825,'b2_bounded_text_composition',
            ['paths','sms_composed','model_sha256'],['paths','sms_composed','grade','csv_sha256']),
    ]


def prediction_cases(work, refs):
    for branch,name,case,features,rows,collection,model_keys,csv_keys in cases(work):
        references = (case/'export_0/rows.csv', case/'export_1/rows.csv')
        yield branch,name,case/'relocated',features,rows,collection,model_keys,csv_keys,references
    if CANONICAL in refs:
        for seed,rows in (('45101',1672), ('45102',1686)):
            case = work/'ab_canonical_output/runs/canonical_preference_01'/seed
            features = work/'ab_contextual_preference/runs/context_headlines_02'/seed/'features.zip'
            references = (case/'native/rows.csv', case/'relocated/rows.csv')
            yield (CANONICAL,'canonical_'+seed,case/'moved',features,rows,'b_canonical_output',
                   [seed,'model_sha256'],[seed,'export_sha256'],references)


def verify(bundle, workspace, seed_repo, output):
    output.mkdir(parents=True,exist_ok=False)
    repo = output/'clone'
    call(['git','init','-q',repo],output)
    call(['git','fetch','--no-tags',seed_repo,BASE+':refs/heads/receiver-baseline'],repo)
    raw = call(['git','bundle','list-heads',bundle],repo)
    refs = {name.removeprefix('refs/heads/'):commit for commit,name in (l.split() for l in raw.splitlines())}
    required = {STABLE,ROUTE,KERNEL,RECIPE}
    if set(refs) not in (required,required|{CANONICAL}):
        raise ValueError('Expected the registered integration and study branches')
    for commit in refs.values():
        if subprocess.run(['git','cat-file','-e',commit+'^{commit}'],cwd=repo,capture_output=True).returncode==0:
            raise ValueError('Receiver already has a bundled branch tip')
    call(['git','bundle','verify',bundle],repo)
    call(['git','fetch','--no-tags',bundle,'refs/heads/*:refs/heads/*'],repo)
    call(['git','fsck','--full','--no-reflogs'],repo)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(map(str,(repo/'scripts/ab_acceptance_guard',repo))),
        AB_B1_OFFLINE='1',AB_B1_FIT_LOG=str(output/'physical_fits.jsonl'),SUPERAGENT_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',
        SUPERAGENT_MINILM_WEIGHTS=str(workspace/'pretrained_weights/all-MiniLM-L6-v2'),
        OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    env.pop('SUPERAGENT_MINILM_CACHE_DIR',None)
    report = dict(bundle_sha256=file_hash(bundle),bundle_bytes=bundle.stat().st_size,receiver_seed_commit=BASE,
        bundle_verified_with_only_seed_history=True,all_tips_absent_before_import=True,fsck='passed',
        branch_tips=refs,python=sys.version.split()[0],predictions={},new_fits=0,new_model_api_requests=0)
    for branch,name,relocated,features,rows,collection,model_keys,csv_keys,references in prediction_cases(workspace,refs):
        call(['git','switch','-q',branch],repo)
        if call(['git','rev-parse','HEAD'],repo)!=refs[branch] or call(['git','status','--porcelain','--untracked-files=no'],repo):
            raise ValueError('Wrong or dirty receiver branch')
        source = call([sys.executable,'-c','from pathlib import Path; import superagent; print(Path(superagent.__file__).resolve())'],repo,env)
        if Path(source)!=repo/'superagent/__init__.py':
            raise ValueError('Prediction imported a workspace checkout outside the clean receiver')
        record = json.loads((repo/'research'/collection/'results.json').read_text())
        model_id = get(record,model_keys)
        export_id = get(record,csv_keys)
        selection = json.loads((relocated/'final_selection.json').read_text())
        if model_id!=selection['model_sha256'] or file_hash(relocated/selection['model_file'])!=model_id:
            raise ValueError('Frozen model does not match the imported receipt')
        if any(file_hash(ref)!=export_id for ref in references):
            raise ValueError('Workspace references differ from the imported receipt')
        expected_input = file_hash(features)
        dest = output/name/'rows.csv'
        log = call([sys.executable,'-m','superagent','predict',features,'--run-dir',relocated,'--output',dest],repo,env)
        (output/(name+'.log')).write_text(log+'\n')
        with dest.open(encoding='utf-8',newline='') as f:
            contents = list(csv.DictReader(f))
        if len(contents)!=rows or file_hash(dest)!=export_id:
            raise ValueError('Receiver changed predictions or row order')
        report['predictions'][name] = dict(branch=branch,rows=rows,candidate=selection['family'],
            model_sha256=model_id,features_sha256=expected_input,csv_sha256=export_id,identical=True)
        if branch==CANONICAL:
            probe = dest.parent/'probability_check.json'
            call([sys.executable,'-m','scripts.check_b_canonical_preference_output','probe',
                  '--model',relocated,'--features',features,'--labels',dest,'--output',probe],repo,env)
            measured = json.loads(probe.read_text())
            recorded = record[model_keys[0]]
            if any(recorded[key]!=value for key,value in measured.items()):
                raise ValueError('Receiver changed adapted probabilities or their recorded diagnostics')
            report['predictions'][name].update(positive_probability_sha256=measured['positive_probability_sha256'],
                max_swap_probability_error=measured['max_swap_probability_error'])
    if (output/'physical_fits.jsonl').exists():
        raise ValueError('A receiver verification unexpectedly fitted a classifier')
    (output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle','workspace','seed-repo','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    verify(args.bundle.resolve(),args.workspace.resolve(),args.seed_repo.resolve(),args.output.resolve())
