"""Reload models exported by the normal CLI on small anonymous classification fixtures."""
import argparse
import csv
import io
import json
from pathlib import Path
import zipfile

import numpy as np

from tools.pretrained_study import read,write
from tools.submission_smoke import make_package,command,check_rows


def csv_package(path, kind):
    rng=np.random.default_rng(1866)
    rows=[]
    classes=['red','green','blue']
    vocabulary=('harbor ferry ticket','garden fruit market','festival stage music')
    for i in range(525):
        value=rng.normal()
        target=(i%3+int(value>=0))%3 if kind=='mixed' else i%3
        label=classes[int(rng.integers(3)) if i%4==0 else target]
        if kind=='mixed':
            rows.append([value,['x','y','z'][i%3],label])
        else:
            rows.append([f'{vocabulary[i%3]} event report {i}',label])
    columns=([dict(name='u',type='numeric'),dict(name='v',type='categorical')] if kind=='mixed'
             else [dict(name='u',type='text')])
    manifest=dict(schema_version=1,modality='tabular' if kind=='mixed' else 'text',
                  metric='accuracy' if kind=='mixed' else 'macro_f1',columns=columns,target='y',
                  splits={s:dict(format='csv',path=s+'.csv') for s in ('train','test')})
    with zipfile.ZipFile(path,'w') as out:
        out.writestr('manifest.json',json.dumps(manifest))
        for split in ('train','test'):
            stream=io.StringIO()
            writer=csv.writer(stream)
            writer.writerow([c['name'] for c in columns]+(['y'] if split=='train' else []))
            writer.writerows(rows[:-15] if split=='train' else [r[:-1] for r in rows[-15:]])
            out.writestr(split+'.csv',stream.getvalue())
    return {'red','green','blue'},15


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    root=p.parse_args().output.resolve()
    root.mkdir(parents=True,exist_ok=True)
    cases={}
    for name,kind in (('a','tabular'),('b','image'),('c','mixed'),('d','text')):
        folder=root/name
        folder.mkdir()
        pkg=folder/'anonymous.zip'
        classes,count=csv_package(pkg,kind) if kind in ('mixed','text') else make_package(pkg,kind,False)
        run=folder/'run'
        command(folder/'training.log','run',pkg,'--run-dir',run,'--method-choice','paired',
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
    write(root/'smoke_summary.json',cases)
    print({k:(v['kind'],v['trials'],v['winner'],v['predictions']) for k,v in cases.items()})


if __name__=='__main__':main()
