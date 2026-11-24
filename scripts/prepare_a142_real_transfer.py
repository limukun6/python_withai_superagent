"""One official public download, grouped holdout registration, no training or model API."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import joblib
import numpy as np
import requests

from scripts.a_batch_common import hashes,save
from scripts.run_a10_real_task_pilot import read,sha

ROOT=Path(__file__).resolve().parents[1]
URL='https://archive.ics.uci.edu/static/public/267/banknote%2Bauthentication.zip'


def packet(path,x,y=None):
    split='train' if y is not None else 'test';buf=io.BytesIO()
    np.savez(buf,**({'X':x,'y':y} if y is not None else {'X':x}))
    manifest=dict(schema_version=1,objective='classification',metric='accuracy',modality='tabular',
                  splits={split:dict(format='npz',path='anonymous.npz')})
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('manifest.json',json.dumps(manifest));z.writestr('anonymous.npz',buf.getvalue())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();source=hashes(ROOT)
    save(out/'protocol.json',dict(batch='A142',source_files=source,runner_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT/'docs/batches/A142_A144_new_numeric_transfer_plan_20261124_zh.md'),
         dataset_page='https://archive.ics.uci.edu/dataset/267/banknote',doi='10.24432/C55P57',
         license='CC BY 4.0',download_url=URL,maximum_public_requests=1,maximum_model_API_requests=0,
         maximum_fits=0,outer_seed=71424,inner_seeds=[707,808,909],primary_metric='accuracy'))
    try:
        proxies={'http':'http://127.0.0.1:7897','https':'http://127.0.0.1:7897'}
        began=time.monotonic();response=requests.get(URL,proxies=proxies,timeout=60,allow_redirects=False)
        response.raise_for_status();body=response.content
        if response.status_code!=200 or len(body)>2**20:raise ValueError('Unexpected public archive response/size')
        (out/'original.zip').write_bytes(body)
        with zipfile.ZipFile(io.BytesIO(body)) as z:
            members=[n for n in z.namelist() if Path(n).name=='data_banknote_authentication.txt']
            if len(members)!=1:raise ValueError('Expected exactly one registered data member')
            raw=z.read(members[0])
        rows=np.loadtxt(io.BytesIO(raw),delimiter=',')
        if rows.shape!=(1372,5) or not np.isfinite(rows).all() or set(rows[:,-1])!={0.,1.}:
            raise ValueError('Official dataset structure differs from registration')
        from scripts import a38_predict_guard
        from superagent.data import TrainingTask,load_training_task
        from superagent.splits import grouped_partition,sample_groups
        from superagent.contracts import TaskSpec
        from superagent.ab_run import registered_capabilities
        from superagent.prediction_evidence import fingerprint
        from tools.ab_registered_rows import content_digest
        x=rows[:,:4];y=rows[:,4].astype(int);original=TrainingTask('numeric_tabular',x,y,{'metric':'accuracy'})
        fit,held,split_info=grouped_partition(original,71424,'shared_text');groups=sample_groups(original,'shared_text')
        if set(groups[fit])&set(groups[held]):raise ValueError('Outer exact-feature group leakage')
        packet(out/'anonymous_train.zip',x[fit],y[fit]);packet(out/'query.zip',x[held])
        private=out/'private';private.mkdir();joblib.dump(dict(X=x[held],y=y[held],row_ids=held.tolist()),private/'holdout.joblib')
        task=load_training_task(out/'anonymous_train.zip');spec=TaskSpec.from_task(task).record()
        caps=registered_capabilities(spec,len(task.y_train),'full');inner=[]
        for seed in (707,808,909):
            train,search,info=grouped_partition(task,seed,'shared_text')
            inner.append(dict(seed=seed,train_indices=train.tolist(),search_indices=search.tolist(),
                              row_ids_sha256=fingerprint(search),label_sha256=fingerprint(task.y_train[search])))
        save(out/'registered_rows.json',dict(version=1,kind=task.kind,grouping='exact_rows',
             training_content_sha256=content_digest(task.X_train,task.y_train),splits=inner))
        counts=Counter(groups.tolist());duplicates=sum(n-1 for n in counts.values())
        results=dict(download=dict(url=URL,status=response.status_code,seconds=time.monotonic()-began,
                         archive_sha256=sha(out/'original.zip'),member=members[0],member_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(body)),
             original_rows=len(y),features=4,train_rows=len(fit),held_rows=len(held),duplicate_feature_rows=duplicates,
             outer_split=split_info,outer_train_row_ids_sha256=fingerprint(fit),outer_held_row_ids_sha256=fingerprint(held),
             exact_feature_group_overlap=0,training_archive_sha256=sha(out/'anonymous_train.zip'),
             query_archive_sha256=sha(out/'query.zip'),holdout_sha256=sha(private/'holdout.joblib'),
             registered_rows_sha256=sha(out/'registered_rows.json'),task_spec=spec,capabilities=caps,
             majority_training_label=int(np.bincount(y[fit]).argmax()),training_class_counts=dict(Counter(map(str,y[fit]))),
             source_files=source,outer_labels_given_to_agent=False,outer_performance_computed=False,
             candidate_policy='Frozen default full catalogue; numeric-kernel and diagnostics off')
        if source!=hashes(ROOT):raise ValueError('Source changed')
        save(out/'results.json',results)
        save(out/'completion.json',dict(status='prepared',actual_classifier_fits=0,real_API_requests=0,
             public_download_requests=1,elapsed_seconds=time.monotonic()-start,source_unchanged=True,
             protocol_sha256=sha(out/'protocol.json'),results_sha256=sha(out/'results.json')))
        print(json.dumps({k:results[k] for k in ('original_rows','train_rows','held_rows','duplicate_feature_rows','exact_feature_group_overlap')}))
    except Exception as e:
        save(out/'completion.json',dict(status='failed',error=str(e),actual_classifier_fits=0,real_API_requests=0,
             automatic_retry=False,elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':main()
