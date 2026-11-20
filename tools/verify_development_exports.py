"""Check exported models on the complete development pool and collect their portable bundles."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import joblib
import numpy as np

from superagent.data import load_training_task
from superagent.encoder_mixture import EncoderMixture
from superagent.minilm_features import MiniLMTextClassifier, feature_usage
from superagent.models import predict_with_model
from superagent.pretrained_choice import digest
from superagent.splits import grouped_partition, take
from tools.frontier_study import expect, receipt
from tools.pretrained_study import read, train_only, write


def using_warm_features(folder, model, kind, rows):
    parts = (model.anchor,model.other) if isinstance(model,EncoderMixture) else (model,)
    for part in parts:
        if isinstance(part, MiniLMTextClassifier):
            part.runtime = str((folder/'pretrained_runtime.json').resolve())
            part.encoder = None
    before = feature_usage(folder)
    predicted = predict_with_model(kind, model, rows)
    expect(feature_usage(folder)==before, 'Export verification unexpectedly added encoder forwards')
    return predicted


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('current','previous','task1','task2','solutions','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    report={}
    for case in ('task1','task2','task2-wide'):
        task_name=case.split('-')[0]
        root=args.current/case
        run=root/'run'
        rec=read(root/'results.json')
        protocol=read(root/'protocol.json')
        archive=getattr(args,task_name)
        expect(digest(archive)==protocol['archive_sha256'],'Wrong task archive')
        opened=[]
        with train_only(archive,opened):
            data=load_training_task(archive)
        keep,reserve,_=grouped_partition(data,protocol['reserve_seed'],'shared_text')
        expect(len(keep)==rec['development_training_rows'] and len(reserve)==rec['reserved_rows'],'Pool changed')
        x=take(data.X_train,keep)
        original=joblib.load(run/'selected_model.joblib')
        export=joblib.load(root/'portable'/'selected_model.joblib')
        expected=using_warm_features(run,original['model'],data.kind,x)
        np.testing.assert_array_equal(expected,using_warm_features(run,export['model'],data.kind,x))
        rec['warm_portable_comparison_rows']=len(expected)
        rec['development_pool_prediction_sha256']=hashlib.sha256(json.dumps(expected.tolist()).encode()).hexdigest()
        rec['training_members']=opened
        if case==task_name:
            preceding=args.previous/case/'run'
            old=receipt(preceding)
            expect(old['winner']==original['candidate'],'Final default changed the selected family')
            previous=joblib.load(preceding/'selected_model.joblib')['model']
            np.testing.assert_array_equal(expected,using_warm_features(preceding,previous,data.kind,x))
            for trial in range(1,8):
                f=f'trial_{trial}.joblib'
                np.testing.assert_array_equal(joblib.load(run/f)['search_predictions'],joblib.load(preceding/f)['search_predictions'])
            rec['previous_default_comparison']=dict(same_primary_prediction_arrays=7,same_pool_predictions=len(expected))
        name='development' if case==task_name else 'development-mixture'
        target=args.solutions/task_name/name
        target.mkdir(exist_ok=True,parents=True)
        selection=read(root/'portable'/'final_selection.json')
        files={selection['model_file'],selection['single_model_file'],'final_selection.json','task_spec.json'}
        for filename in sorted(files):
            src=root/'portable'/filename
            shutil.copyfile(src,target/filename)
            expect(digest(src)==digest(target/filename),'Copied file changed')
        expect(digest(target/selection['model_file'])==selection['model_sha256']==rec['portable_model_sha256'],
               'Export does not match its manifest')
        rec['portable_files']={file:digest(target/file) for file in sorted(files)}
        rec['solution_directory']=str(Path('solutions')/task_name/name)
        report[case]=rec
    write(args.output,dict(tasks=report,scope='Fits on the evaluated development pool only. Portable and original predictions agree on every development row and on independent synthetic examples. These equivalence checks supply no additional accuracy estimate.'))
    print({k:(v['selected'],v['warm_portable_comparison_rows']) for k,v in report.items()},flush=True)


if __name__=='__main__':
    main()
