"""Transfer the two frozen UCI movie-holdout winners to the whole public Cornell v1.0 corpus."""
import argparse
from collections import Counter
import csv
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import numpy as np

from scripts import run_b1_ab_acceptance as record
from scripts.audit_b_kernel_grouped_effect import grouped
from scripts.run_b_c1_matrix import read_export
from scripts.run_b_kernel_domain_transfer import read_original
from scripts.run_b_nonlinear_text import identity, pair_grade
from scripts.run_bq_01_portfolios import pack
from superagent import ab_run
from superagent.data import load_development_task, load_prediction_features
from superagent.splits import normalized_text, sample_groups
from tools.ab_registered_rows import content_digest


SOURCE_URL = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
ARCHIVE = 'a05befe52aafda71d458d188a1c54506a998b1308613ba76bbda2e5029409ce9'
MEMBERS = {
    'rt-polaritydata/rt-polarity.neg': ('negative','4af43bf489620b498a0852c5a8c6a535155dec7985fdfea782b8f7756b957ccc'),
    'rt-polaritydata/rt-polarity.pos': ('positive','d79ef14009bbf87cd6be21f91c3ea1a3cbcb73b34dfb4231aef7bbcd7677abc4'),
}
FROZEN = {'original':'46a85166d1a25f0abad0bc36caaed9a67d371ea5ea3a6e31863043c0d3c6d428',
          'extended':'8ae5466eb0fbefa09320a74e13453eda3fbe10cc6d6362342bec5c41a05ce464'}


def external_input(path, uci):
    if ab_run.sha(path)!=ARCHIVE:
        raise ValueError('External movie-review archive differs from the original publisher download')
    seen = {normalized_text(t) for data in read_original(uci).values() for t,_ in data}
    kept,encodings = [],{}
    excluded = Counter()
    with tarfile.open(path,'r:gz') as archive:
        for member,(label,expected) in MEMBERS.items():
            source = archive.getmember(member)
            if not source.isfile():
                raise ValueError('Unexpected public dataset member')
            raw = archive.extractfile(source).read()
            if record.sha(raw)!=expected:
                raise ValueError('Public polarity member has changed')
            try:
                text = raw.decode('utf-8')
                encodings[member] = 'utf-8'
            except UnicodeDecodeError:
                text = raw.decode('latin-1')
                encodings[member] = 'latin-1'
            rows = text.removesuffix('\n').split('\n')
            if len(rows)!=5331 or any(not row.strip() for row in rows):
                raise ValueError('Unexpected number of nonempty released snippets')
            for i,text in enumerate(rows):
                text = text.removesuffix('\r')
                if normalized_text(text) in seen:
                    excluded[label]+=1
                else:
                    kept.append((record.sha(text.encode('utf-8')),member,i,text,label))
    kept.sort(key=lambda item:item[:3])
    return [r[3] for r in kept],np.asarray([r[4] for r in kept]),dict(encodings=encodings,
        excluded_uci_exact_rows=dict(excluded),original_rows=10662,
        order_sha256=record.digest([[r[0],r[1],r[2]] for r in kept]))


def study(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    record.offline(out)
    roots = dict(original=args.control.resolve(),extended=record.ROOT)
    previous = args.previous.resolve()
    signed = record.ROOT/'research/b_text_kernel_domains'
    observed_source = identity(roots['original'])
    for name in ('protocol','results','completion','all_six_models_frozen'):
        if ab_run.sha(previous/(name+'.json'))!=ab_run.sha(signed/(name+'.json')):
            raise ValueError('Frozen UCI study differs from published source')
    if observed_source!=record.read(signed/'protocol.json')['sources']:
        raise ValueError('Original or extended model implementation changed')
    record.write(out/'protocol.json',dict(batch='B-TEXT-KERNEL-07',published_source=SOURCE_URL,
        plan_sha256=ab_run.sha(record.ROOT/'docs/batches/B_TEXT_KERNEL_07_cornell_transfer_plan.md'),
        driver_sha256=ab_run.sha(Path(__file__)),archive_sha256=ARCHIVE,members=MEMBERS,models=FROZEN,
        bootstrap_seed=112204,bootstrap_draws=4000,new_fit_limit=0,model_api_limit=0))
    try:
        models = {}
        for arm,expected in FROZEN.items():
            folder = previous/('imdb_'+arm)/'relocated'
            selection = record.read(folder/'final_selection.json')
            if (selection['model_sha256']!=expected or ab_run.sha(folder/selection['model_file'])!=expected
                    or record.read(signed/'all_six_models_frozen.json')['imdb_'+arm]!=expected):
                raise ValueError('Target model is not the preregistered, frozen decision')
            models[arm] = folder
        rows,labels,metadata = external_input(args.cornell,args.uci)
        packet = out/'features.zip'
        pack(packet,'single_text',rows,None,'macro_f1','test')
        task = load_development_task(previous/'imdb/train.zip')
        if (content_digest(rows,labels)!=content_digest(load_prediction_features(packet,task.kind,task.spec),labels)
                or set(labels.tolist())!=set(task.y_train.tolist())):
            raise ValueError('External label semantics or feature order changed')
        record.write(out/'input_and_models_frozen.json',dict(**metadata,kept_rows=len(rows),
            class_counts=dict(Counter(labels.tolist())),model_sha256=FROZEN,features_sha256=ab_run.sha(packet),
            held_content_sha256=content_digest(rows,labels)))
        files = {}
        for arm,root in roots.items():
            env = dict(os.environ,PYTHONPATH=os.pathsep.join(map(str,(root/'scripts/ab_acceptance_guard',root))),
                SUPERAGENT_OFFLINE='1',SUPERAGENT_MINILM_WEIGHTS=str(args.weights.resolve()),
                OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
            env.pop('SUPERAGENT_MINILM_CACHE_DIR',None)
            dest = out/arm/'rows.csv'
            cmd = [sys.executable,'-m','superagent','predict',str(packet),'--run-dir',str(models[arm]),'--output',str(dest)]
            with (out/(arm+'.log')).open('x') as log:
                subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
            read_export(dest,len(rows))
            files[arm] = dest
        record.write(out/'prediction_files_frozen.json',{arm:ab_run.sha(f) for arm,f in files.items()})
        grades,delta = pair_grade(labels,task,files)
        predictions = [np.asarray(read_export(f,len(rows)),dtype=labels.dtype) for f in files.values()]
        groups = sample_groups(SimpleNamespace(kind='single_text',X_train=rows))
        uncertainty = grouped(labels,predictions,groups,'macro_f1',grades)
        by_class = {k:dict(rows=int((labels==k).sum()),original_recall=float((predictions[0][labels==k]==k).mean()),
            extended_recall=float((predictions[1][labels==k]==k).mean())) for k in task.spec.get('labels',np.unique(labels).tolist())}
        if record.trace(out):
            raise ValueError('Unexpected fit while using a frozen model')
        record.write(out/'results.json',dict(grades=grades,paired=delta,by_class=by_class,grouped_uncertainty=uncertainty))
        record.write(out/'completion.json',dict(status='completed',new_fits=0,cli_prediction_runs=2,held_rows=len(rows),
            new_model_api_requests=0,new_course_rmb=0,official_validation_test_or_H=False))
    except Exception as error:
        record.write(out/'completion.json',dict(status='failed',reason=str(error),new_fits=len(record.trace(out))))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('control','previous','cornell','uci','weights','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    study(parser.parse_args())
