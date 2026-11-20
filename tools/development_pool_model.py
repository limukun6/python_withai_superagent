"""Fit the released-training development pool and check a movable inference model on synthetic input."""
import argparse
import csv
import io
import os
from pathlib import Path
import shutil
import zipfile

import joblib
import numpy as np

from superagent.benchmark import run_one
from superagent.data import DevelopmentTask, load_training_task
from superagent.minilm_features import MiniLMTextClassifier
from superagent.pretrained_choice import digest
from superagent.splits import grouped_partition, take
from tools.frontier_study import budget_ids, expect, receipt
from tools.method_device_study import OPTIONS
from tools.pretrained_study import read, train_only, write
from tools.soft_vote_diagnostic import freeze_file
from tools.submission_smoke import command


def synthetic_package(path, data):
    with zipfile.ZipFile(path, 'w') as out:
        if data.kind == 'categorical_tabular':
            n_columns = data.X_train.shape[1]
            rows = [[f'synthetic-feature-{j}-{i%3}' for j in range(n_columns)] for i in range(12)]
            buffer = io.BytesIO()
            np.save(buffer,np.asarray(rows),allow_pickle=False)
            out.writestr('C_test.npy', buffer.getvalue())
        elif data.kind == 'pair_text':
            buffer = io.StringIO()
            writer = csv.writer(buffer)
            writer.writerow(['headline_A','headline_B'])
            for word in ('bridge','pool','platform','library','workshop','tram'):
                pair = (f'The city {word} opens next month', f'A first look at the new {word}')
                writer.writerows([pair,pair[::-1]])
            out.writestr('test.csv', buffer.getvalue())
        else:
            raise ValueError('No synthetic portability fixture for this modality')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('archive','reference','output','ledger','historical-protocol','weights','minilm-weights'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    os.environ['SUPERAGENT_OFFLINE']='1'
    ref = read(args.reference/'protocol.json')
    expect(ref['archive_sha256']==digest(args.archive) and ref['options']==OPTIONS,'Training configuration differs')
    repo = Path(__file__).resolve().parents[1]
    expect(all(digest(repo/name)==sha for name,sha in ref['source'].items()),'Default policy source has changed')
    opened=[]
    with train_only(args.archive,opened):
        task=load_training_task(args.archive)
    keep,reserve,_=grouped_partition(task,ref['reserve_seed'],'shared_text')
    previously_used=set(i for split in ref['indices'] for i in split['outer'])
    expect(set(keep)==previously_used and not set(keep)&set(reserve) and len(reserve)==ref['reserved'],
           'Final development fit would change the reserved population')
    folder=args.output.resolve()
    folder.mkdir(parents=True,exist_ok=True)
    run=folder/'run'
    protocol=dict(reference_protocol_sha256=digest(args.reference/'protocol.json'),archive_sha256=ref['archive_sha256'],
                  source_sha256=digest(Path(__file__)),search_seed=ref['search_seed'],reserve_seed=ref['reserve_seed'],
                  rows=len(keep),reserved=len(reserve),accessed=opened,
                  rule='Use the unchanged auto policy on all previously used development rows. No new performance estimate. Predict only on newly generated, unlabeled input to compare original and portable models.')
    freeze_file(folder/'protocol.json',protocol)
    if args.freeze:return
    task=DevelopmentTask(task.kind,'training_development_pool',take(task.X_train,keep),task.y_train[keep],None,None,task.spec)
    roots=[Path(p) for p in ref['shared_budget_roots']]+[folder.parent]
    previous=next(t for t in read(args.historical_protocol)['tasks'] if t['package_sha256']==ref['archive_sha256'])
    config=dict(ledger=str(args.ledger.resolve()),scope=previous['scope'],minilm_weights=str(args.minilm_weights.resolve()),
                tabular_weights=str(args.weights.resolve()),cap_rmb=ref['arm_cap'],campaign_cap_rmb=ref['campaign_cap'],
                deferred_ids=previous['legacy_unknown_ids'],campaign_run_roots=list(map(str,roots)),
                previous_run_ids=budget_ids(args.ledger,previous['scope'],roots,[run]))
    if not (run/'summary.json').exists():
        expect(not run.exists(),'Incomplete earlier run needs separate accounting')
        run_one(task,'local_interleaved_routes',ref['search_seed'],run,None,**OPTIONS,pretrained_config=config,method_choice='auto')
    rec=receipt(run)
    portable=folder/'portable'
    portable.mkdir(exist_ok=True)
    selection=read(run/'final_selection.json')
    expect(selection['training_rows']==len(keep) and selection['refit'],'Full development refit missing')
    model_name=selection['model_file']
    bundle=joblib.load(run/model_name)
    model=bundle['model']
    if isinstance(model,MiniLMTextClassifier):
        model.runtime='pretrained_runtime.json'
        model.encoder=None
        joblib.dump(bundle,portable/model_name)
    else:
        shutil.copyfile(run/model_name,portable/model_name)
    shutil.copyfile(run/'task_spec.json',portable/'task_spec.json')
    selection['model_sha256']=digest(portable/model_name)
    write(portable/'final_selection.json',selection)
    package=folder/'synthetic.zip'
    synthetic_package(package,task)
    checks=[]
    for name,source in (('original',run),('portable',portable)):
        output=folder/(name+'.csv')
        command(folder/(name+'.log'),'predict',package,'--run-dir',source,'--minilm-weights',args.minilm_weights.resolve(),'--output',output)
        manifest=read(folder/'prediction_manifest.json')
        expect(manifest['prediction_count']==12 and not manifest['test_labels_read'],'Synthetic input check failed')
        checks.append(digest(output))
    expect(checks[0]==checks[1],'The moved model predicted differently')
    report=dict(development_training_rows=len(keep),reserved_rows=len(reserve),selected=rec['winner'],trial_6_7=rec['extras'],
                fits=rec['fits'],generation_calls=rec['calls'],local_qwen_input_tokens=rec['qwen_tokens'],
                minilm_training_input_tokens=rec['minilm_tokens'],course_reference_rmb=rec['course_rmb'],api_rmb=rec['api_rmb'],
                original_model_sha256=rec['model_sha256'],portable_model_sha256=selection['model_sha256'],
                synthetic_prediction_sha256=checks[0],synthetic_unlabeled_rows=12,
                development_protocol_sha256=digest(folder/'protocol.json'),course_validation_or_test_opened=False)
    write(folder/'results.json',report)
    print(report,flush=True)


if __name__=='__main__':
    main()
