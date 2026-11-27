"""Offline local-checkpoint TabPFN screen, using fixed training-only subsets."""
import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',DO_NOT_TRACK='1',
                  TABPFN_DISABLE_TELEMETRY='1',TABPFN_ALLOW_CPU_LARGE_DATASET='1')

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import OrdinalEncoder

from scripts.run_task1_profile_screen import digest, load_split, save
from superagent.joint_context import blocks, tokens
from superagent.splits import sample_groups

WEIGHT_SHA = 'cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49'


def run(task,out):
    out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    def offline(event,args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):
            raise PermissionError('Network disabled for the local TabPFN screen')
    sys.addaudithook(offline)
    torch.set_num_threads(6)
    checkpoint=Path(os.environ['APPDATA'])/'tabpfn/tabpfn-v2-classifier-finetuned-zk73skhh.ckpt'
    fits=0
    members=0
    rows=[]
    try:
        if digest(checkpoint.read_bytes())!=WEIGHT_SHA:
            raise ValueError('Local weight identity does not match the official file')
        from tabpfn import TabPFNClassifier
        from importlib.metadata import version
        x,y,hashes=load_split(task,'train')
        groups=sample_groups(SimpleNamespace(kind='categorical_tabular',X_train=x))
        splits=list(StratifiedGroupKFold(5,shuffle=True,random_state=18643).split(x,y,groups))[:2]
        root=Path(__file__).resolve().parents[1]
        refs=__import__('json').loads((root/'research/task1_joint_context_01/results.json').read_text(encoding='utf-8'))
        save(out/'protocol.json',dict(members=hashes,weight=checkpoint.name,weight_sha=WEIGHT_SHA,
            tabpfn=version('tabpfn'),torch=torch.__version__,device='cpu',max_training_rows=2048,
            n_estimators=2,classifier_member_cap=8,seed=26471,
            runner_sha=digest(Path(__file__).read_bytes()),
            model_sources={p.name:digest(p.read_bytes()) for p in (root/'superagent/joint_context.py',
                           root/'superagent/profile_history.py',root/'superagent/splits.py')},
            baseline_results_sha=digest((root/'research/task1_joint_context_01/results.json').read_bytes()),
            validation_opened=False,test_opened=False,gpu_probe_error='no kernel image available for sm120'))
        for fold,(train,held) in enumerate(splits):
            chosen,_=train_test_split(train,train_size=2048,stratify=y[train],random_state=26471)
            reference=refs[fold]
            if digest(held.astype('<i8').tobytes()) != __import__('json').loads(
                (root/'research/task1_joint_context_01/protocol.json').read_text(encoding='utf-8'))['splits'][fold]['held_sha']:
                raise ValueError('Reference evaluation rows differ')
            row=dict(fold=fold,train_rows=len(chosen),held_rows=len(held),
                     chosen_sha=digest(chosen.astype('<i8').tobytes()),held_sha=digest(held.astype('<i8').tobytes()))
            for name in ('raw','with_context'):
                tick=time.monotonic()
                a,b=x[chosen],x[held]
                if name=='with_context':
                    profile,context=blocks(x[train])
                    a=np.column_stack((a,tokens(a[:,profile]),tokens(a[:,context])))
                    b=np.column_stack((b,tokens(b[:,profile]),tokens(b[:,context])))
                encoder=OrdinalEncoder(dtype=np.float32,handle_unknown='use_encoded_value',unknown_value=-1)
                a=encoder.fit_transform(a)
                b=encoder.transform(b)
                model=TabPFNClassifier(n_estimators=2,auto_scale_n_estimators=False,device='cpu',
                    categorical_features_indices=list(range(a.shape[1])),model_path=checkpoint,
                    fit_mode='fit_preprocessors',random_state=26471,n_preprocessing_jobs=1)
                fits+=1
                members+=2
                save(out/'last_attempt.json',dict(fold=fold,method=name,fit_calls=fits,admitted_members=members))
                model.fit(a,y[chosen])
                if model.n_estimators_!=2:
                    raise ValueError('Unexpected TabPFN ensemble member count')
                probabilities=np.concatenate([model.predict_proba(b[i:i+128])[:,1] for i in range(0,len(b),128)])
                if list(model.classes_)!=[0,1] or not np.isfinite(probabilities).all():
                    raise ValueError('Invalid class order or probabilities')
                correct=int(np.sum((probabilities>.5)==y[held]))
                accuracy=correct/len(held)
                row[name]=dict(correct=correct,accuracy=accuracy,seconds=round(time.monotonic()-tick,3),
                               versus_tokens_pp=100*(accuracy-reference['tokens']['accuracy']))
                print(fold,name,row[name],flush=True)
                save(out/'results.json',rows+[row])
            rows.append(row)
        summary={}
        for name in ('raw','with_context'):
            deltas=[r[name]['versus_tokens_pp'] for r in rows]
            summary[name]=dict(mean_delta_pp=float(np.mean(deltas)),promote=bool(min(deltas)>0 and np.mean(deltas)>=.5))
        save(out/'completion.json',dict(status='complete',fit_api_calls=fits,classifier_members=members,
            api_calls=0,tokens=0,course_yuan=0,seconds=round(time.monotonic()-start,3),summary=summary))
    except Exception as exc:
        save(out/'completion.json',dict(status='failed',error=repr(exc),fit_api_calls=fits,
             admitted_classifier_members=members,seconds=round(time.monotonic()-start,3),api_calls=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    run(args.task,args.output)
