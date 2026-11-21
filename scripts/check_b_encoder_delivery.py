"""Test frozen prediction with copied, missing, and wrong-checksum local encoder assets."""
import argparse
import csv
import os
from pathlib import Path
import shutil
import subprocess
import sys

from scripts import run_b1_ab_acceptance as fixture
from superagent.ab_sentence import FILES, REVISION


def weights(source, root):
    good, bad = root/'pinned_files', root/'changed_copy'
    good.mkdir()
    bad.mkdir()
    for name,(size,digest) in FILES.items():
        local = good/name
        shutil.copy2(source/name,local)
        if local.stat().st_size != size or fixture.sha(local.read_bytes()) != digest:
            raise ValueError('Local resource differs from registered revision: '+name)
        if name=='config.json':
            changed = bytearray(local.read_bytes())
            changed[0] ^= 1
            (bad/name).write_bytes(changed)
        else:
            os.link(local,bad/name)
    return {'verified':good, 'absent':None, 'modified':bad}


def predict_case(label, path, source, reference, arm, root, settings):
    dest = root/label
    dest.mkdir()
    record = fixture.read(reference/'results.json')['paths']
    if isinstance(record,list):
        record = {p['name']:p for p in record}
    expected = record[arm]
    entry = fixture.read(reference/'protocol.json')['tasks'][arm]
    chosen = fixture.read(path/'run'/'final_selection.json')
    for filename in ('task_spec.json','final_selection.json',chosen['model_file']):
        shutil.copy2(path/'run'/filename,dest/filename)
    model_hash = fixture.sha((dest/chosen['model_file']).read_bytes())
    if model_hash != expected['model_sha256'] or model_hash != chosen['model_sha256']:
        raise ValueError('Copied model differs from the previously frozen export')
    feature_file = path/'features.zip'
    if fixture.sha(feature_file.read_bytes()) != entry['files']['features.zip']:
        raise ValueError('Unlabeled input differs from the frozen feature packet')
    original = expected['delivery']['csv_sha256']
    rows = expected['delivery']['rows']
    outcomes = {}
    for case, folder in settings.items():
        case_dir = dest/case
        cache = case_dir/'empty_cache'
        cache.mkdir(parents=True)
        env = dict(os.environ, HF_HOME=str(cache), HF_HUB_CACHE=str(cache/'hub'), TRANSFORMERS_OFFLINE='1',
                   HF_HUB_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
        env.pop('SUPERAGENT_MINILM_WEIGHTS',None)
        if folder is not None:
            env['SUPERAGENT_MINILM_WEIGHTS'] = str(folder)
        output = case_dir/'rows.csv'
        proc = subprocess.run([sys.executable,'-m','superagent','predict',str(feature_file),
            '--run-dir',str(dest),'--output',str(output)], env=env, cwd=case_dir,
            stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True, timeout=300)
        (case_dir/'predict.log').write_text(proc.stdout,encoding='utf-8')
        if source=='lexical' or case=='verified':
            if proc.returncode or not output.is_file() or fixture.sha(output.read_bytes())!=original:
                raise ValueError(f'Frozen export changed: {label}/{case}')
            with output.open(newline='',encoding='utf-8') as stream:
                ids = [item['row_id'] for item in csv.DictReader(stream)]
            if ids != [str(i) for i in range(rows)]:
                raise ValueError('Frozen export row order changed')
            outcomes[case] = dict(result='identical_export', rows=rows, csv_sha256=original)
        else:
            if (proc.returncode==0 or output.exists() or (case_dir/'prediction_manifest.json').exists()
                    or 'Set SUPERAGENT_MINILM_WEIGHTS' not in proc.stdout):
                raise ValueError('Missing or changed assets did not produce the expected explicit rejection')
            outcomes[case] = dict(result='rejected_resource_before_export', exit_code=proc.returncode)
    return dict(model_sha256=model_hash, family=chosen['family'], results=outcomes)


def check(args):
    root = args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    fixture.offline(root)
    plan = [
        ('preference_encoder',args.q.resolve(),'encoder','bq_01_portfolios','headlines_expanded'),
        ('symmetric_encoder',args.mrpc.resolve(),'encoder','b1_06_mrpc','45101_extended'),
        ('symmetric_lexical',args.mrpc.resolve(),'lexical','b1_06_mrpc','45101_lexical')]
    fixture.write(root/'protocol.json',dict(batch='BD-01', reference_paths=[p[-1] for p in plan], revision=REVISION,
        files=FILES, cases=['verified','absent','modified'], driver_sha256=fixture.sha(Path(__file__).read_bytes()),
        plan_sha256=fixture.sha((fixture.ROOT/'docs/batches/BD_01_plan.md').read_bytes()), fits_allowed=0, model_api_allowed=0))
    results = {}
    try:
        settings=weights(args.weights,root)
        for label,parent,source,reference,arm in plan:
            results[label]=predict_case(label,parent/arm,source,fixture.ROOT/'research'/reference,arm,root,settings)
            fixture.write(root/'results.json',results)
        fixture.write(root/'completion.json',dict(status='passed', fresh_processes=9, identical_exports=5,
            expected_resource_rejections=4, new_training=0, external_model_requests=0, fresh_scores=0))
    except Exception as error:
        fixture.write(root/'completion.json',dict(status='failed', reason=str(error), completed=list(results),
            new_training=0, external_model_requests=0, fresh_scores=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('q','mrpc','weights','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    check(parser.parse_args())
