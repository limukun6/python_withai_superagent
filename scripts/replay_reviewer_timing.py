"""Compare the real parent and revised review scheduling using recorded training measurements."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
from pathlib import Path
import shutil
import subprocess
import time
from unittest.mock import patch

import joblib
import numpy as np

from scripts.replay_joint_error_groups import observations
from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent import task_opening
from superagent.data import load_training_task
from superagent.events import RunLogger
from superagent.metrics import MetricSpec

PROTOCOL=ROOT/'research/joint_feedback_1127/reviewer_timing_protocol.json'


def original(stage):
    base=read(PROTOCOL)['parent']
    blob=subprocess.check_output(['git','show',base+':superagent/task_opening.py'],cwd=ROOT)
    filename=stage/'parent_opening.py'
    filename.write_bytes(blob)
    spec=importlib.util.spec_from_file_location('superagent.recorded_parent_opening',filename)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def replay(module,folder,source,reply_kind):
    folder.mkdir()
    q=read(source/'task_opening.json')
    incumbent=read(source/'opening_incumbent.json')
    task=load_training_task(source.parents[2]/'task.zip',read_declarations=True)
    metric=MetricSpec.for_task(task)
    folds={}
    payloads={}
    observed={}
    target=q['tested_candidate']
    for fold in q['confirmation']:
        part=read(source/(fold+'_split.json'))
        inputs=source/(fold+'.joblib')
        ids,bank=observations(source,fold,task,metric,target)
        bank.pop(target,None)  # It is measured by the two reserved replay operations below.
        folds[fold]=(np.asarray(part['fit']),ids,{})
        payloads[fold]=dict(path=inputs,sha256=sha(inputs),labels=task.y_train[ids])
        observed[fold]=bank
    start=time.monotonic()
    opening=module.TaskOpening('opening-feedback-flash',task,metric,folds,[],'ordinary',RunLogger(folder),None,start,60)
    opening.brief=read(source/'opening_brief.json')
    opening.answer=q['local_hypothesis']
    opening.record.update(local_hypothesis=opening.answer,interpretation_status='pending')
    response=dict(opening.answer,check='keep')
    if reply_kind=='late_valid':
        alternatives=opening.brief['checks']
        key=next(k for k in sorted(alternatives) if k!=q['check'] and alternatives[k]['kind']=='slice')
        response.update(check=key,citations=list(dict.fromkeys([*response['citations'],alternatives[key]['source']])))
    def reply():
        if reply_kind=='ready':return response
        time.sleep(2.5)
        if reply_kind=='late_valid':return response
        raise RuntimeError('Controlled timeout after a single request')
    opening.pool=ThreadPoolExecutor(max_workers=1)
    opening.future=opening.pool.submit(reply)
    if reply_kind=='ready':opening.future.result(timeout=2)
    marks=[]
    class FrozenExecutor:
        limit=11
        caps={target:{'classifier_fits':1,'refit_classifier_fits':1}}
        def accounting(self):return {'attempted_classifier_fits':q['fits_at_delivery']+len(marks),'unknown_failed_attempts':0}
        def worker(self,payload,candidate,artifact,timeout,reserve):
            require(candidate==target,'Probe diverged from the frozen admissible alternative')
            fold=Path(payload).stem
            marks.append(dict(fold=fold,pending=not opening.future.done(),seconds=round(time.monotonic()-decision_start,3)))
            existing=source/('opening_probe_'+fold+'.joblib')
            shutil.copyfile(existing,artifact)
            return None,joblib.load(artifact)['metrics']
    menu=[dict(action='explore',members=[target])]
    choice={'members':q['incumbent_members']}
    decision_start=time.monotonic()
    with patch.object(module,'actions',return_value=menu):
        try:
            result,_=opening.after_delivery({},choice,incumbent,[],observed,payloads,{},FrozenExecutor(),60,set())
        finally:
            opening.finish()
    require(len(marks)==2 and result['model_sha256']==incumbent['model_sha256'],'Model or matched operations changed')
    return dict(probes=marks,source_at_decision=opening.record.get('hypothesis_source_at_decision'),
                status_at_decision=opening.record['decision_time_status'],status_at_close=opening.record['interpretation_status'],
                late_lesson=opening.record.get('late_hypothesis'),
                model_sha=result['model_sha256'],result=opening.record['result'],elapsed=round(time.monotonic()-decision_start,3))


def main(root, late_only=False):
    dst=root/'timing-replay'
    source=root/'development/d58/feedback-local/run/agent'
    if late_only:
        run=replay(task_opening,dst/'late-valid',source,'late_valid')
        prior=read(dst/'verified.json')['revised']
        require(run['status_at_decision']=='pending' and run['status_at_close']=='valid'
                and run['source_at_decision']=='local' and run['model_sha']==prior['model_sha']
                and run['late_lesson']['used_for_model'] is False, 'Late evidence rewrote the frozen decision')
        save(dst/'late_verified.json',dict(fixture='delayed successful answer with a different check',replay=run,
                                          model_api_requests=0,physical_fits=0))
        return
    dst.mkdir()
    old=original(dst)
    before=replay(old,dst/'parent',source,'delayed_error')
    after=replay(task_opening,dst/'revised',source,'delayed_error')
    ready=replay(task_opening,dst/'ready',source,'ready')
    require(not before['probes'][0]['pending'] and after['probes'][0]['pending']
            and before['status_at_close']==after['status_at_close']=='invalid' and ready['source_at_decision']=='campus',
            'Timing or settlement expectation failed')
    require(before['model_sha']==after['model_sha']==ready['model_sha'],'Model changed during timing replay')
    save(dst/'verified.json',dict(protocol=sha(PROTOCOL),parent=before,revised=after,ready_reply=ready,
                                 physical_fits=0,model_api_requests=0,outer_data_access=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=lambda s:Path(s).resolve())
    parser.add_argument('--late-only',action='store_true')
    args=parser.parse_args()
    main(args.root,args.late_only)
