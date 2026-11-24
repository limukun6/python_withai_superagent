"""Recover prediction delivery from read-only run artifacts; never resume training or requests."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_bytes())
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def files(root):return {p.name:digest(p) for p in sorted(root.iterdir()) if p.is_file()}


def recover(source,output,source_root,task_archive):
    source=Path(source).resolve();output=Path(output).resolve();source_root=Path(source_root).resolve()
    task_archive=Path(task_archive).resolve();started=time.monotonic()
    if source==output or output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('Recovery output must be separate from the original run')
    output.mkdir(parents=True,exist_ok=False);before=files(source)
    write(output/'recovery_protocol.json',dict(original_run=str(source),original_files=before,
         requested_source_root=str(source_root),task_archive_sha256=digest(task_archive),
         new_classifier_fit_cap=0,new_API_cap=0,training_resume_supported=False))
    try:
        manifest=read(source/'run_manifest.json')
        actual={p.relative_to(source_root/'superagent').as_posix():digest(p)
                for p in sorted((source_root/'superagent').rglob('*.py'))}
        if actual!=manifest['source_files']:raise ValueError('Exact original training source is required')
        if digest(task_archive)!=manifest['task_archive_sha256']:raise ValueError('Original training archive digest differs')
        sys.path.insert(0,str(source_root))
        from scripts import a38_predict_guard  # no fits or network, including joblib prediction
        import joblib
        from superagent.ab_run import sha
        from superagent.data import load_training_task
        from superagent.contracts import TaskSpec
        from superagent.metrics import MetricSpec
        from superagent.events import RunLogger
        from superagent.partial_delivery import freeze_checkpoint
        from superagent.prediction_evidence import fingerprint
        from superagent.splits import sample_groups
        from tools.ab_registered_rows import validated_ids
        if not Path(sys.modules['superagent'].__file__).resolve().is_relative_to(source_root):
            raise ValueError('Wrong training source was imported')
        receipts=[];unknown=[]
        for path in sorted(source.glob('*.fits.json')):
            try:
                item=read(path);n=item.get('attempted_classifier_fits');done=item.get('completed_classifier_fits')
                known=item.get('status')=='completed' and type(n) is int and type(done) is int and n==done and n>=1
            except (ValueError,OSError):item={};known=False
            receipts.append(dict(file=path.name,sha256=digest(path),status=item.get('status','unreadable'),
                                 attempted=n if known else None,completed=done if known else None))
            if not known:unknown.append(path.name)
        for path in source.glob('*.admission.json'):
            if not path.with_name(path.name.replace('.admission.json','.fits.json')).is_file():unknown.append(path.name)
        ledger=dict(inherited_worker_receipts=receipts,incomplete_or_unknown_worker_files=sorted(set(unknown)),
                    known_completed_classifier_fits=sum(r['completed'] or 0 for r in receipts),
                    additional_past_fit_cost_unknown=bool(unknown),new_classifier_fits=0,new_API_requests=0,
                    training_resume_supported=False,original_global_API_ledger_modified=False)
        write(output/'inherited_resources.json',ledger)
        shutil.copy2(source/'run_manifest.json',output/'run_manifest.json')
        logger=RunLogger(output);logger.initialize_usage_ledger()
        if (source/'final_selection.json').exists():
            selected=read(source/'final_selection.json');name=selected['model_file']
            if Path(name).name!=name:raise ValueError('Frozen model filename escapes original run')
            if digest(source/name)!=selected['model_sha256'] or digest(source/'task_spec.json')!=selected['task_spec_sha256']:
                raise ValueError('Frozen model or task schema digest differs')
            for name in ('final_selection.json','task_spec.json',name):shutil.copy2(source/name,output/name)
            result=dict(mode='existing_frozen_selection',delivery_status=selected.get('delivery_status','complete' if selected['refit'] else 'partial'),
                        refit_performed=False,original_refit_status_preserved=selected['refit'])
        else:
            task=load_training_task(task_archive);spec=TaskSpec.from_task(task).record()
            if spec!=manifest['task_contract']:raise ValueError('Task contract differs from original run')
            candidates=sorted(source.glob('state_*.json'),key=lambda p:int(p.stem.split('_')[1]),reverse=True)
            state=None;skipped=[]
            for path in candidates:
                try:current=read(path)
                except (ValueError,OSError):skipped.append(path.name);continue
                history=current.get('history',[])
                if not history:continue
                if current.get('task')!=spec or [r['trial'] for r in history]!=list(range(1,len(history)+1)):
                    raise ValueError('Parsed state has conflicting task or trial order')
                state=current;state_path=path;break
            if state is None:raise ValueError('No complete nonempty state checkpoint is available')
            trusted={};groups=sample_groups(task,'shared_text');metric=MetricSpec.for_task(task)
            for split in ('primary','confirmation_1','confirmation_2'):
                record=read(source/(split+'_split.json'));payload_path=source/(split+'.joblib')
                if digest(payload_path)!=record['evaluation_payload_sha256']:raise ValueError('Partition payload digest differs')
                fit,held=validated_ids(record,task.y_train,groups);payload=joblib.load(payload_path)
                if (payload['metric_spec']!=metric.record() or
                    fingerprint(payload['y_search'])!=fingerprint(task.y_train[held])):
                    raise ValueError('Partition metric/labels differ from registered task')
                trusted[split]=dict(row_ids=held.tolist(),labels=task.y_train[held].tolist(),groups=groups[held].tolist(),
                                    evaluation_payload_sha256=sha(payload_path))
                if payload.get('preference_vote_probabilities'):
                    trusted[split]['tie_label_one']=[a<b for a,b in payload['X_search']]
                for name in (split+'.joblib',split+'_split.json'):shutil.copy2(source/name,output/name)
            for row in state['history']:
                for suffix in ('.joblib','.fits.json'):
                    name='trial_'+str(row['trial'])+suffix;shutil.copy2(source/name,output/name)
            proof=freeze_checkpoint(task,state,trusted,manifest['capabilities'],output,logger,
                 'Explicit recovery of a verified checkpoint after interruption; no new training or requests')
            result=dict(mode='verified_partial_checkpoint',**proof,state_file=state_path.name,
                        state_sha256=digest(state_path),ignored_incomplete_states=skipped,refit_performed=False)
        if files(source)!=before:raise ValueError('Original source files changed during recovery')
        write(output/'recovery_completion.json',dict(status='recovered',**result,
             new_classifier_fits=0,new_API_requests=0,training_resume_supported=False,
             original_files_unchanged=True,elapsed_seconds=time.monotonic()-started,
             inherited_resource_receipt_sha256=digest(output/'inherited_resources.json')))
        return read(output/'recovery_completion.json')
    except Exception as e:
        write(output/'recovery_completion.json',dict(status='rejected',error=str(e),new_classifier_fits=0,new_API_requests=0,
              original_files_unchanged=files(source)==before,training_resume_supported=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','output','source-root','task-archive'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(recover(a.run_dir,a.output,a.source_root,a.task_archive)))
