"""Rebuild paired accuracy, model traces and resource totals from the training-only artifacts."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import joblib
import numpy as np

from superagent.data import load_training_task
from superagent.models import predict_with_model
from superagent.pretrained_choice import digest
from superagent.pricing import resolve_pricing
from superagent.splits import effective_grouping,sample_groups,take
from tools.frontier_study import expect,receipt
from tools.pretrained_study import read,write,train_only


def paired_summary(difference,groups):
    mean=float(np.mean(difference))
    _,members=np.unique(groups,return_inverse=True)
    counts=np.bincount(members)
    sums=np.bincount(members,weights=difference)
    error=math.sqrt(len(counts)/(len(counts)-1)*np.square(sums-mean*counts).sum())/len(difference)
    return dict(points=100*mean,conditional_95_points=[100*(mean-1.96*error),100*(mean+1.96*error)],
                feature_groups=len(counts))


def resource_totals(root,ledger):
    complete=partial=finished_fits=partial_fits=local_tokens=batches=mini_tokens=mini_batches=api_calls=0
    event_prices={}
    entries=[]
    for path in sorted(root.glob('*/fold_*/*/trials.json')):
        folder=path.parent
        summary=folder/'summary.json'
        if (folder/'frontier_receipt.json').exists():
            rec=receipt(folder)
            complete+=1;finished_fits+=rec['fits'];api_calls+=rec['calls']
            expect(rec['api_rmb']==0,'Unexpected service debit')
        else:
            trials=read(path)
            if summary.exists():
                run=read(summary)
                fits=run['classifier_fit_accounting']['completed_classifier_fits']
                api_calls+=run['llm_calls']
                expect(run['actual_cost_rmb']==0,'Unexpected service debit on unfinished run')
            else:
                fits=sum(t.get('classifier_fits',0) for t in trials if t['status']=='completed')
                if (folder/'confirmation.json').exists():
                    fits+=read(folder/'confirmation.json').get('classifier_fits_completed',0)
            partial_fits+=fits;partial+=1
            entries.append(dict(stage=folder.parent.parent.name,fold=folder.parent.name,arm=folder.name,fits=fits))
        for file in (folder/'llm_usage.jsonl',folder/'downstream_usage.jsonl'):
            if not file.exists():continue
            for line in file.read_text().splitlines():
                event=json.loads(line)
                if event['event']=='local_forward':
                    key=event['request_id']
                    price=resolve_pricing(event['model']).course_cost(event['input_tokens'],event['output_tokens'])
                    expect(key not in event_prices and event['output_tokens']==0 and event['actual_cost_rmb']==0
                           and abs(event['course_cost_rmb']-price)<1e-10,'Reused or unexpected local event')
                    event_prices[key]=round(price*1e9)
                    local_tokens+=event['input_tokens'];batches+=1
                elif event['event']=='frozen_feature_forward':
                    expect(event['remote_calls']==event['api_rmb']==0,'Unexpected MiniLM service charge')
                    mini_tokens+=event['input_tokens'];mini_batches+=1
    with sqlite3.connect(f'file:{ledger.resolve()}?mode=ro',uri=True) as conn:
        rows=conn.execute("SELECT id,status,course_units,json_extract(payload,'$.run_id') FROM requests")
        known=[(key,status,cost) for key,status,cost,run in rows if run and Path(run).resolve().is_relative_to(root)]
    expect({k:c for k,_,c in known}==event_prices and all(s=='known' for _,s,_ in known),'Usage log and ledger differ')
    rmb=sum(c for _,_,c in known)/1e9
    expect(api_calls==0,'Unexpected generation request')
    inference_tokens=inference_batches=0
    for log in root.glob('*/fold_*/*_prediction_features/downstream_usage.jsonl'):
        for line in log.read_text().splitlines():
            event=json.loads(line)
            expect(event['event']=='frozen_feature_forward' and event['remote_calls']==event['api_rmb']==0,'Invalid scoring event')
            inference_tokens+=event['input_tokens']
            inference_batches+=1
    return dict(physical_complete_runs=complete,complete_fits=finished_fits,partial_runs=partial,partial_fits=partial_fits,
                partial_details=entries,qwen_input_tokens=local_tokens,qwen_batches=batches,qwen_reference_rmb=rmb,
                minilm_training_input_tokens=mini_tokens,minilm_training_batches=mini_batches,
                minilm_scoring_input_tokens=inference_tokens,minilm_scoring_batches=inference_batches,
                generation_calls=api_calls,api_rmb=0.)


def diagnostic_summary(folder,protocol,output,reference,task):
    result={k:v for k,v in output.items() if k not in ('folds','training_members')}
    folds=[]
    for i,(split,summary) in enumerate(zip(reference['indices'],output['folds'],strict=True)):
        path=folder/f'fold_{i}'
        rows=read(path/'outer_predictions.json')
        held=split['outer']
        expect(rows['rows']==held,'Diagnostic rows changed')
        truth=task.y_train[held]
        np.testing.assert_array_equal(truth,rows['truth'])
        if 'predictions' in rows:
            counts=[]
            for k,vector in enumerate(rows['predictions'],1):
                model_file=path/f'model_{k}.joblib'
                expect(digest(model_file)==read(path/f'model_{k}.json')['sha256'],'Learning-curve fit changed')
                predicted=predict_with_model(task.kind,joblib.load(model_file),take(task.X_train,held))
                np.testing.assert_array_equal(predicted,vector)
                counts.append(int(np.sum(predicted==truth)))
        else:
            counts={key:int(np.sum(np.asarray(rows[key])==truth)) for key in summary['correct']}
            if 'full_finalists' in rows:
                best=max(np.sum(np.asarray(v)==truth) for v in rows['full_finalists'].values())
                expect(best==summary['oracle_single'],'Wrong shortlist upper bound')
        expect(counts==summary['correct'] and summary['rows']==len(truth),'Incorrect diagnostic fold count')
        folds.append({k:v for k,v in summary.items() if k in ('correct','rows','adopted','oracle_single','train_counts')})
    agg=([sum(f['correct'][i] for f in folds) for i in range(len(folds[0]['correct']))]
         if isinstance(folds[0]['correct'],list) else {k:sum(f['correct'][k] for f in folds) for k in folds[0]['correct']})
    expect(agg==output['correct'] and sum(f['rows'] for f in folds)==output['rows'],'Incorrect diagnostic totals')
    return dict(result,folds=folds,seed=reference['seed'],protocol_sha256=digest(folder/'protocol.json'),
                results_sha256=digest(folder/'results.json'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('root','task1','task2','ledger','output'):p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--require',nargs='*',default=[])
    args=p.parse_args()
    root=args.root.resolve()
    tasks={}
    for archive in (args.task1,args.task2):
        opened=[]
        with train_only(archive,opened):task=load_training_task(archive)
        groups=sample_groups(task,effective_grouping(task.kind,'shared_text'))
        tasks[digest(archive)]=(task,groups,opened)
    cache={};stages={};repeats={};diagnostics={}
    protocols={digest(path):read(path) for path in root.glob('*/protocol.json')}
    for file in sorted(root.glob('*/results.json')):
        folder=file.parent
        protocol=read(folder/'protocol.json')
        output=read(file)
        if 'locations' not in protocol:
            ref=protocols[protocol.get('reference_sha256',protocol.get('reference'))]
            diag_task=tasks[ref['archive_sha256']][0]
            diagnostics[folder.name]=diagnostic_summary(folder,protocol,output,ref,diag_task)
            continue
        task,groups,_=tasks[protocol['archive_sha256']]
        by_arm={k:[] for k in output['correct']}
        rowids=[]
        per_fold=[]
        for i,(split,summary) in enumerate(zip(protocol['indices'],output['folds'],strict=True)):
            outer=read(folder/f"fold_{protocol['fold_numbers'][i]-1}"/'outer_predictions.json')
            ids=split['outer'];labels=task.y_train[ids]
            expect(outer['rows']==ids and not set(ids).intersection(rowids),'Mismatched or repeated outer rows')
            np.testing.assert_array_equal(outer['truth'],labels)
            rowids.extend(ids)
            first=None
            fold_correct={}
            for arm,rec in summary['arms'].items():
                source=Path(protocol['locations'][i][arm])
                key=(str(source),tuple(ids))
                if key not in cache:
                    expect(digest(source/'selected_model.joblib')==rec['model_sha256'],'Selected model changed')
                    bundle=joblib.load(source/'selected_model.joblib')
                    usage_dir=source.parent/(arm+'_prediction_features')
                    from superagent.minilm_features import MiniLMTextClassifier,feature_usage
                    cached=isinstance(bundle['model'],MiniLMTextClassifier)
                    if cached:
                        before=feature_usage(usage_dir)
                        config=read(usage_dir/'pretrained_runtime.json')
                        bundle['model'].bind_prediction_weights(config[bundle['model'].weight_setting],usage_dir)
                    predicted=predict_with_model(task.kind,bundle['model'],take(task.X_train,ids))
                    if cached:expect(before==feature_usage(usage_dir),'Audit encountered an uncached evaluation feature')
                    initial=[]
                    for k in range(1,6):
                        result=joblib.load(source/f'trial_{k}.joblib')['search_predictions'].tolist()
                        initial.append(hashlib.sha256(json.dumps(result,ensure_ascii=False).encode()).hexdigest())
                    cache[key]=(predicted,initial)
                predicted,initial=cache[key]
                np.testing.assert_array_equal(predicted,outer[arm])
                if first is None:first=initial
                else:expect(initial==first,'Early search differed row by row')
                correct=predicted==labels
                expect(int(correct.sum())==rec['correct'],'Incorrect published count')
                fold_correct[arm]=int(correct.sum())
                by_arm[arm].append(correct)
            per_fold.append(fold_correct)
        rowids=np.asarray(rowids);order=np.argsort(rowids);rowids=rowids[order]
        expect(len(rowids)==output['rows'],'Wrong pooled row count')
        masks={k:np.concatenate(v)[order].astype(int) for k,v in by_arm.items()}
        baseline=next(iter(masks))
        comparisons={}
        for arm,mask in masks.items():
            expect(int(mask.sum())==output['correct'][arm],'Wrong pooled accuracy')
            if arm==baseline:continue
            diff=mask-masks[baseline]
            deltas=[f[arm]-f[baseline] for f in per_fold]
            comparisons[arm]=dict(**paired_summary(diff,groups[rowids]),corrected=int((diff>0).sum()),lost=int((diff<0).sum()),
                                  fold_deltas=deltas,wins=sum(x>0 for x in deltas),ties=deltas.count(0),losses=sum(x<0 for x in deltas))
            repeat=repeats.setdefault((protocol['archive_sha256'],baseline,arm),{})
            entry=(rowids.copy(),diff)
            if protocol['seed'] in repeat:
                original=repeat[protocol['seed']]
                expect(np.array_equal(original[0],entry[0]) and np.array_equal(original[1],entry[1]),'Conflicting results for the same seed')
            else:repeat[protocol['seed']]=entry
        traces={name:[dict(winner=f['arms'][name]['winner'],trial_6_7=f['arms'][name]['extras']) for f in output['folds']]
                for name in by_arm}
        stages[folder.name]=dict(seed=protocol['seed'],rows=len(rowids),correct=output['correct'],baseline=baseline,traces=traces,
                                 comparisons=comparisons,qualified=output['qualified'],fold_correct=per_fold,
                                 protocol_sha256=digest(folder/'protocol.json'),results_sha256=digest(file))
    expect(set(args.require)<=stages.keys(),'A required comparison is absent')
    repeated=[]
    for (archive,base,arm),seeds in repeats.items():
        if len(seeds)<2:continue
        indices=next(iter(seeds.values()))[0]
        if any(not np.array_equal(ids,indices) for ids,_ in seeds.values()):continue
        avg=np.mean([d for _,d in seeds.values()],axis=0)
        repeated.append(dict(scope=tasks[archive][0].kind,baseline=base,arm=arm,seeds=sorted(seeds),unique_rows=len(avg),
                             **paired_summary(avg,tasks[archive][1][indices])))
    resources=resource_totals(root,args.ledger)
    resources['extra_diagnostic_fits']=sum(r.get('diagnostic_fits',r.get('additional_diagnostic_fits',0)) for r in diagnostics.values())
    device_selections={}
    for file in root.glob('*/fold_*/*/trials.json'):
        for trial in read(file):
            metadata=trial.get('training_metadata') or {}
            choice=metadata.get('device_decision')
            if choice:
                device=choice['selected']
                device_selections[device]=device_selections.get(device,0)+1
    resources['primary_scratch_device_choices']=device_selections
    report=dict(stages=stages,diagnostics=diagnostics,repeated_seeds=repeated,unique_model_reloads=len(cache),resources=resources,
                training_members={t[0].kind:t[2] for t in tasks.values()},
                interval_scope='Approximate group-robust intervals conditional on these fitted predictions. Repeated seeds average correctness differences within each row before estimating uncertainty.')
    write(args.output,report)
    print('verified',len(stages),'stages and',len(cache),'models',report['resources'],flush=True)


if __name__=='__main__':main()
