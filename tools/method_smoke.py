"""Reload models exported by the normal CLI on small anonymous classification fixtures."""
import argparse
import csv
import io
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np

from tools.pretrained_study import read,write
from tools.submission_smoke import make_package,command,check_rows
from superagent.method_choice import MODES


def csv_package(path, kind):
    rng=np.random.default_rng(1866)
    rows=[]
    classes=['red','green','blue']
    vocabulary=('harbor ferry ticket','garden fruit market','festival stage music')
    for i in range(225 if kind=='categories' else 525):
        value=rng.normal()
        target=(i%3+int(value>=0))%3 if kind=='mixed' else i%3
        label=classes[int(rng.integers(3)) if i%4==0 else target]
        if kind=='categories':
            rows.append([vocabulary[i%3].split()[0],rng.choice(['east','west','north','south']),
                         f'group-{int(rng.integers(32))}',label])
        elif kind=='mixed':
            rows.append([value,['x','y','z'][i%3],label])
        elif kind=='pair':
            left=f'{vocabulary[i%3]} early edition {i}'
            right=f'{vocabulary[(i+1)%3]} late edition {i}'
            left_wins=i%2==0
            rows.append([left,right,str(int(left_wins))] if i%4<2 else [right,left,str(int(not left_wins))])
        else:
            rows.append([f'{vocabulary[i%3]} event report {i}',label])
    if kind=='categories':
        columns=[dict(name=c,type='categorical') for c in ('u','v','w')]
    elif kind=='mixed':
        columns=[dict(name='u',type='numeric'),dict(name='v',type='categorical')]
    elif kind=='pair':
        columns=[dict(name=c,type='text') for c in ('u','v')]
    else:
        columns=[dict(name='u',type='text')]
    manifest=dict(schema_version=1,modality='text' if kind=='text' else 'tabular',
                  metric='macro_f1' if kind=='text' else 'accuracy',columns=columns,target='y',
                  splits={s:dict(format='csv',path=s+'.csv') for s in ('train','test')})
    if kind=='pair':
        manifest.update(modality='pair_text',pair_label_one='left')
    with zipfile.ZipFile(path,'w') as out:
        out.writestr('manifest.json',json.dumps(manifest))
        for split in ('train','test'):
            stream=io.StringIO()
            writer=csv.writer(stream)
            writer.writerow([c['name'] for c in columns]+(['y'] if split=='train' else []))
            writer.writerows(rows[:-15] if split=='train' else [r[:-1] for r in rows[-15:]])
            out.writestr(split+'.csv',stream.getvalue())
    return ({'0','1'} if kind=='pair' else {'red','green','blue'}),15


def check_relocated_text_model(folder,weights):
    moved=folder/'relocated'
    moved.mkdir()
    bundle=moved/'model'
    bundle.mkdir()
    for name in ('selected_model.joblib','final_selection.json','task_spec.json'):
        shutil.copyfile(folder/'run'/name,bundle/name)
    output=moved/'predictions.csv'
    records=[]
    for attempt in ('fresh','cached'):
        command(moved/(attempt+'.log'),'predict',folder/'anonymous.zip','--run-dir',bundle,
                '--minilm-weights',Path(weights).resolve(),'--output',output)
        assert output.read_bytes()==(folder/'predictions.csv').read_bytes()
        usage=read(moved/'prediction_manifest.json')['prediction_feature_usage']
        assert usage['remote_calls']==usage['api_rmb']==0
        records.append(usage)
    assert records[0]['input_tokens']>0 and records[1]['input_tokens']==0
    write(moved/'checks.json',dict(first=records[0],cached=records[1],byte_identical=True))
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--minilm-weights',type=Path,help='Also exercise real offline frozen features through the CLI without a ledger or credentials')
    p.add_argument('--policy',choices=MODES,help='Use the same named method policy on all anonymous fixtures')
    args=p.parse_args()
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=True)
    cases={}
    fixtures=[('a','tabular'),('b','image'),('c','mixed'),('d','text')]
    if args.minilm_weights:fixtures.append(('e','categories'))
    if args.policy:fixtures.append(('f','pair'))
    for name,kind in fixtures:
        folder=root/name
        folder.mkdir()
        pkg=folder/'anonymous.zip'
        classes,count=make_package(pkg,kind,False) if kind in ('tabular','image') else csv_package(pkg,kind)
        run=folder/'run'
        flags=[]
        if args.minilm_weights and (kind=='categories' or args.policy and kind in ('text','pair')):
            settings=folder/'features.json'
            write(settings,dict(minilm_weights=str(args.minilm_weights.resolve())))
            flags=['--pretrained-config',settings]
        policy=args.policy or ('fixed' if flags else 'paired')
        command(folder/'training.log','run',pkg,'--run-dir',run,'--method-choice',policy,*flags,
                '--planner','local','--trials',7,'--feature-cache','run','--time-limit',600,'--trial-timeout',90)
        outcome=read(run/'run_outcome.json')
        assert outcome['requested_planner_completed'] and outcome['new_api_requests']==0
        assert not outcome['formal_test_accessed'] and not outcome['provided_validation_available']
        output=folder/'predictions.csv'
        command(folder/'predict.log','predict',pkg,'--run-dir',run,'--output',output)
        with output.open() as f:check_rows(list(csv.DictReader(f)),classes,count)
        manifest=read(folder/'prediction_manifest.json')
        assert not manifest['test_labels_read'] and manifest['prediction_count']==count
        cases[name]=dict(kind=outcome['task_kind'],metric=outcome['primary_metric'],predictions=count,
                         trials=outcome['primary_trials_attempted'],stop=outcome['search_stop_reason'],
                         winner=outcome['selected_candidate'],api_calls=outcome['new_api_requests'],
                         initial_protected=read(run/'method_policy.json')['protected_initial_trials'],
                         decisions=[read(f) for f in sorted(run.glob('method_decision_*.json'))])
        feature_trace=run/'downstream_usage.jsonl'
        tried_mini=any(t['candidate'].startswith('minilm_') and t['status']=='completed' for t in read(run/'trials.json'))
        if flags and not args.policy:
            assert tried_mini
        if tried_mini:
            forwards=[json.loads(line) for line in (run/'downstream_usage.jsonl').read_text().splitlines()]
            assert forwards and all(r['remote_calls']==r['api_rmb']==0 for r in forwards)
            cases[name]['minilm_input_tokens']=sum(r['input_tokens'] for r in forwards)
            cases[name]['minilm_forwards']=len(forwards)
            cases[name]['minilm_api_rmb']=sum(r['api_rmb'] for r in forwards)
        elif feature_trace.exists():
            raise ValueError('Feature usage exists without a successful MiniLM trial')
        if args.policy and outcome['selected_candidate'] in ('minilm_text_with_lexical_features','minilm_text_dense_only') and args.minilm_weights:
            cases[name]['relocated_prediction_usage']=check_relocated_text_model(folder,args.minilm_weights)
    write(root/'smoke_summary.json',cases)
    print({k:(v['kind'],v['trials'],v['winner'],v['predictions']) for k,v in cases.items()})


if __name__=='__main__':main()
