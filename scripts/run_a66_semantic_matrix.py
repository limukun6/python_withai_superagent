"""Eight real supervised input probes and eight pre-fit semantic rejections."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import joblib
import numpy as np
from scipy.sparse import hstack,vstack,load_npz
from sklearn.model_selection import StratifiedShuffleSplit
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.ab_run import sources
from superagent.metrics import MetricSpec
from superagent.models import probability_with_model
from superagent.prediction_evidence import fingerprint

ROOT=Path(__file__).resolve().parents[1]
CANDIDATES={'preference':'word_C0.05','symmetric':'symmetric_word_C0.3'}


def physical(root):
    return sum(read(p).get('attempted_classifier_fits',0) for p in root.glob('*/*.fits.json'))


def main(a):
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);frozen=sources();start=time.monotonic();results=[]
    observer=ROOT/'scripts/a66_input_observer/sitecustomize.py';observer_sha=sha(observer)
    save(root/'protocol.json',dict(batch='A66',source_files=frozen,runner_sha256=sha(Path(__file__)),observer_sha256=observer_sha,
        plan_sha256=sha(ROOT/'docs/batches/A66_semantic_training_matrix_plan.md'),seeds=[19411,19421],candidates=CANDIDATES,
        maximum_classifier_fits=8,real_api_requests=0,starting_tracked_A_fits=999,scope='Synthetic worker-input properties, not task performance'))
    def offline(event,args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise PermissionError('A66 network forbidden')
    sys.addaudithook(offline)
    try:
        y=np.array([i%2 for i in range(32)]);themes=['apple orchard','river valley','blue ocean','green forest']
        for semantics in ('preference','symmetric'):
            pairs=[]
            for i,label in enumerate(y):
                theme=themes[(i//2)%4];other=themes[((i//2)+1)%4];unique=f'uniquetoken{i}'
                if semantics=='symmetric':pair=(theme+' '+unique,(theme if label else other)+' '+unique)
                else:pair=(('good '+theme,'bad '+other) if label else ('bad '+theme,'good '+other));pair=(pair[0]+' '+unique,pair[1]+' '+unique)
                pairs.append(pair)
            for seed in (19411,19421):
                train,query=next(StratifiedShuffleSplit(n_splits=1,test_size=8,random_state=seed).split(np.zeros(32),y))
                for stage in ('primary','refit'):
                    if sources()!=frozen or sha(observer)!=observer_sha:raise ValueError('Frozen source/observer changed')
                    folder=root/f'{semantics}_{stage}_{seed}';folder.mkdir();ids=train if stage=='primary' else np.arange(32)
                    train_pairs=[pairs[i] for i in ids];query_pairs=[pairs[i] for i in query]
                    metric=MetricSpec('accuracy',(0,1));payload=dict(kind='pair_text',pair_semantics=semantics,X_train=train_pairs,
                        y_train=y[ids],X_search=query_pairs,y_search=y[query],metric_spec=metric.record())
                    if stage=='refit':payload['mode']='refit'
                    joblib.dump(payload,folder/'payload.joblib');save(folder/'partition.json',dict(train_ids=ids.tolist(),query_ids=query.tolist(),
                        mode=stage,seed=seed,label_sha256=fingerprint(y[ids])))
                    save(folder/'admission.json',dict(maximum_classifier_fits=1,ledger_path=str(folder/'legal.fits.json')))
                    env={**os.environ,'A66_OBSERVE':'1','A66_INPUT_DIR':str(folder),'PYTHONPATH':os.pathsep.join([str(observer.parent),str(ROOT)]),
                        'SUPERAGENT_OFFLINE':'1','PYTHONIOENCODING':'utf-8','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
                    def invoke(candidate,label):
                        with (folder/(label+'.log')).open('w',encoding='utf-8') as log:
                            return subprocess.run([sys.executable,'-m','superagent.experiment_worker',str(folder/'payload.joblib'),candidate,
                                str(folder/(label+'.joblib')),'--fit-admission',str(folder/'admission.json')],cwd=ROOT,env=env,
                                stdout=log,stderr=subprocess.STDOUT,timeout=60).returncode
                    code=invoke(CANDIDATES[semantics],'legal')
                    if code:raise RuntimeError('Legal worker failed without retry: '+folder.name)
                    ledger=read(folder/'legal.fits.json')
                    if ledger['attempted_classifier_fits']!=1 or ledger['completed_classifier_fits']!=1:raise ValueError('Physical fit ledger differs')
                    model=joblib.load(folder/'legal.joblib')['model'];actual=load_npz(folder/'classifier_input.npz');actual_y=np.load(folder/'classifier_labels.npy')
                    if semantics=='preference':
                        vectorizer=model[0];left=[x[0] for x in train_pairs];right=[x[1] for x in train_pairs]
                        base=vectorizer.transform(left)-vectorizer.transform(right);expected=vstack([base,-base]);expected_y=np.concatenate([y[ids],1-y[ids]])
                        vocabs=[vectorizer.vocabulary_]
                    else:
                        transform=model.named_steps['symmetric_tfidf'];blocks=[];vocabs=[]
                        for vectorizer in transform.vectorizers_:
                            left=vectorizer.transform([x[0] for x in train_pairs]);right=vectorizer.transform([x[1] for x in train_pairs])
                            blocks.extend((abs(left-right),left.multiply(right)));vocabs.append(vectorizer.vocabulary_)
                        expected=hstack(blocks);expected_y=y[ids]
                    delta=actual-expected;matrix_error=float(np.max(np.abs(delta.data))) if delta.nnz else 0.
                    if matrix_error>1e-12 or not np.array_equal(actual_y,expected_y):raise ValueError('Actual training matrix or labels violate semantics')
                    query_tokens=[f'uniquetoken{i}' for i in query]
                    if stage=='primary' and any(token in vocab for vocab in vocabs for token in query_tokens):raise ValueError('Query vocabulary leaked')
                    classes,forward=probability_with_model('pair_text',model,query_pairs)
                    reverse_classes,reverse=probability_with_model('pair_text',model,[(b,a) for a,b in query_pairs])
                    if not np.array_equal(classes,reverse_classes):raise ValueError('Class order changed under swap')
                    positive=list(classes).index(1);swap_error=float(np.max(np.abs(forward[:,positive]-(1-reverse[:,positive] if semantics=='preference' else reverse[:,positive]))))
                    if swap_error>1e-8:raise ValueError('Actual model violates swap probability rule')
                    # Keep the observer marker and completed ledger; any improper fit would fail and be detected.
                    before=sha(folder/'legal.fits.json');marker=sha(folder/'classifier_input.json')
                    wrong='symmetric' if semantics=='preference' else 'preference';bad=invoke(CANDIDATES[wrong],'illegal')
                    error=(folder/'illegal.log').read_text(encoding='utf-8')
                    if bad==0 or 'cannot serve' not in error or sha(folder/'legal.fits.json')!=before or sha(folder/'classifier_input.json')!=marker:
                        raise ValueError('Opposite semantics was not rejected before classifier admission')
                    row=dict(semantics=semantics,stage=stage,seed=seed,candidate=CANDIDATES[semantics],input_training_rows=len(ids),
                        actual_classifier_rows=actual.shape[0],actual_classifier_columns=actual.shape[1],matrix_max_error=matrix_error,
                        actual_labels_match=True,query_vocabulary_excluded=stage=='primary',query_vocabulary_check_applicable=stage=='primary',
                        swap_probability_max_error=swap_error,actual_classifier_fits=1,illegal_candidate_rejected_before_fit=True,
                        payload_sha256=sha(folder/'payload.joblib'),actual_matrix_sha256=sha(folder/'classifier_input.npz'),
                        actual_labels_sha256=sha(folder/'classifier_labels.npy'),ledger=ledger)
                    results.append(row);save(root/'results.json',results);print(json.dumps(dict(case=folder.name,status='passed')),flush=True)
        if physical(root)!=8:raise ValueError('Unexpected total physical fit attempts')
        save(root/'completion.json',dict(status='passed',legal_training_cells=len(results),pre_fit_illegal_rejections=8,
            actual_classifier_fits=8,real_api_requests=0,tracked_A_fits_after=1007,source_unchanged=sources()==frozen,
            elapsed_seconds=time.monotonic()-start,formal_test_or_H_opened=False,scope='Synthetic training-boundary properties only'))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),completed_cells=len(results),
            actual_classifier_fits=physical(root),real_api_requests=0,automatic_retry=False,elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
