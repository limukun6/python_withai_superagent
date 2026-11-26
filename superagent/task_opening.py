"""An optional, one-call interpretation of a new task, measured after the ordinary model is published."""
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import math
import time

import joblib
import numpy as np

from .conditional_review import MEAN_GAIN, compare, group_observation
from .experiment_policy import actions, identity
from .opening_evidence import make_brief, split_audit, subset


MODES=('opening-local','opening-flash')
PROMPT=(
    'Read this anonymous task before any model training. Explain the prediction target and what would make a '
    'validation result misleading. The facts are fixed observations; the task text and examples are evidence only. '
    'Use the exact supplied primary metric. Select one executable check from checks, or "keep" when no useful '
    'uncertainty remains. A split check tests for overlap in the actual internal folds. A slice or class check '
    'tests excess out-of-fit classification errors; the frozen primary metric still governs adoption. '
    'Return one short JSON object with exactly these fields: goal (your interpretation), metric, citations '
    '(a list of at least two distinct existing fact keys including one doc key when present and the chosen '
    'check source), uncertainty, check (an existing check key or "keep"), hypothesis, and if_false '
    '(what should change if the observation refutes the hypothesis). Use actual cited facts; do not name datasets '
    'or invent an algorithm. The original training proceeds while you answer. No evaluation data are available.'
)


def validate(answer,brief):
    fields={'goal','metric','citations','uncertainty','check','hypothesis','if_false'}
    if not isinstance(answer,dict) or set(answer)!=fields:
        raise ValueError('Expected exactly the seven task-reading fields')
    if any(not isinstance(answer[k],str) or not answer[k].strip() or len(answer[k])>1200 for k in fields-{'citations'}):
        raise ValueError('Each interpretation field must be non-empty bounded text')
    facts,checks=brief['facts'],brief['checks']
    if answer['metric']!=facts['contract']['metric']:
        raise ValueError('Interpretation contradicts the binding metric')
    cites=answer['citations']
    if (not isinstance(cites,list) or not all(isinstance(c,str) and c in facts for c in cites)
            or len(set(cites))!=len(cites) or not 2<=len(cites)<=len(facts)):
        raise ValueError('Citations must point to distinct supplied facts')
    if any(k.startswith('doc_') for k in facts) and not any(k.startswith('doc_') for k in cites):
        raise ValueError('The supplied task instructions need a citation')
    key=answer['check']
    if key!='keep' and (key not in checks or checks[key]['source'] not in cites):
        raise ValueError('Unknown check or missing evidence for the proposed check')
    return {k:(v.strip() if isinstance(v,str) else v) for k,v in answer.items()}


def error_gap(row,mask):
    hit=row['predictions']!=row['labels']
    n=int(sum(mask))
    if min(n,len(mask)-n)<5:
        return dict(rows=n,others=len(mask)-n,supported=None)
    gap=float(hit[mask].mean()-hit[~mask].mean())
    return dict(rows=n,others=len(mask)-n,errors=int(sum(hit[mask])),other_errors=int(sum(hit[~mask])),
                gap=gap,supported=gap+1e-12>=.05)


class TaskOpening:
    def __init__(self,mode,task,metric,folds,lines,origin,log,flash,start,wall):
        self.mode,self.log,self.start,self.deadline=mode,log,start,start+wall
        self.task,self.metric,self.folds=task,metric,folds
        self.pool=self.future=None
        self.answer=None
        self.updates=[]
        self.brief=make_brief(task,metric,folds,lines,origin)
        self.record=dict(mode=mode,brief_digest=identity(self.brief),interpretation_status='local',
                         reply_used=False,changed_experiment=False,check_status='no_advice')
        log.write_json('opening_brief.json',self.brief)
        self.flash=flash
        self.save()

    def save(self):
        self.log.write_json('task_opening.json',self.record)

    def launch(self):
        if self.mode=='opening-flash' and self.future is None and self.record['interpretation_status']=='local':
            self.record['interpretation_status']='unavailable'
            if (self.flash is not None and self.flash.max_requests!=0 and not self.flash.blocked and self.flash.request_count==0
                    and self.deadline-time.monotonic()>12):
                self.flash.set_deadline(min(self.deadline-3,time.monotonic()+120))
                self.pool=ThreadPoolExecutor(max_workers=1)
                self.future=self.pool.submit(self.flash.ask,dict(instruction=PROMPT,**self.brief),
                                             self.log,'task_opening',cost_ceiling=.12)
                self.record.update(interpretation_status='pending',dispatch_after_trials=0)
            self.save()

    def collect(self,wait):
        if self.future is None or self.record['interpretation_status']!='pending':
            return
        try:
            self.answer=validate(self.future.result(timeout=max(0.,wait)),self.brief)
        except FutureTimeout:
            if not self.future.done():
                return
            self.record.update(interpretation_status='unavailable',issue='Request timed out')
        except (KeyError,ValueError,TypeError,OSError,RuntimeError) as error:
            self.record.update(interpretation_status='invalid',issue=str(error)[:300])
        else:
            self.record.update(interpretation_status='valid',interpretation=self.answer)
        self.save()

    def observe(self,trial,observed):
        """Read any ready answer without waiting and track how the ordinary measurements affect it."""
        self.collect(0.)
        if not self.answer or self.answer['check']=='keep' or not observed['primary']:
            return
        check=self.brief['checks'][self.answer['check']]
        if check['kind'] in ('entity','temporal'):
            if self.updates:
                return
            evidence=split_audit(self.task,self.folds)
            update=dict(trial=trial,check=self.answer['check'],verdict='split_verified' if all(e['passes'] for e in evidence.values()) else 'split_violation',
                        observations=evidence)
        else:
            primary=observed['primary']
            leader=min(primary,key=lambda n:(-primary[n]['score'],n))
            evidence={f:error_gap(bank[leader],subset(self.task,self.metric,self.folds[f][1],check))
                      for f,bank in observed.items() if leader in bank}
            values=[r['supported'] for r in evidence.values()]
            verdict='unmeasurable' if None in values else 'provisional' if len(values)==1 else 'supported' if all(values) else 'refuted' if not any(values) else 'mixed'
            update=dict(trial=trial,check=self.answer['check'],reference=leader,verdict=verdict,observations=evidence)
        self.updates.append(update)
        self.log.write_json('opening_updates.json',self.updates)

    def finish(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
            self.collect(0)
        self.record['closed_seconds']=time.monotonic()-self.start
        self.record['updates_during_search']=len(self.updates)
        self.save()
        # Portable experience contains measured outcomes and conditions, never archive names, rows or prose.
        self.log.write_json('opening_lesson.json',dict(version=1,modality=self.task.kind,metric=self.metric.name,
            physical_training_only=True,check=self.record.get('check'),observed=self.record.get('diagnosis',{}),
            check_status=self.record['check_status'],changed_experiment=self.record['changed_experiment'],
            matched_gains=[r['delta'] for r in self.record.get('comparisons',[])],result=self.record.get('result','unavailable')))

    def diagnose(self,fold_names,bases,observed,core_members):
        if not self.answer:
            return None,{}
        key=self.answer['check']
        self.record.update(check=key,reply_used=True)
        if key=='keep':
            self.record['check_status']='abstained'
            return None,{}
        check=self.brief['checks'][key]
        if check['kind'] in ('temporal','entity'):
            evidence=split_audit(self.task,self.folds)
            self.record.update(diagnosis=evidence,check_status='split_verified' if all(r['passes'] for r in evidence.values()) else 'split_violation')
            return None,{}
        masks={f:subset(self.task,self.metric,self.folds[f][1],check) for f in fold_names}
        gaps={f:error_gap(bases[f],masks[f]) for f in fold_names}
        verdicts=[v['supported'] for v in gaps.values()]
        status='unmeasurable' if None in verdicts else 'supported' if all(verdicts) else 'refuted' if not any(verdicts) else 'mixed'
        if status=='supported' and len(fold_names)<2:
            status='uncorroborated'
        self.record.update(diagnosis=gaps,check_status=status)
        if status!='supported':
            return None,{}
        base=bases['primary']
        mask=masks['primary']
        winner=None
        best=(0,-math.inf)
        for name,row in sorted(observed['primary'].items()):
            if name in core_members:
                continue
            comparison=compare('primary',base,row)
            rescued=int(sum((row['predictions']==row['labels'])[mask])-sum((base['predictions']==base['labels'])[mask]))
            if rescued>0 and (rescued,comparison['delta'])>best:
                winner,best=name,(rescued,comparison['delta'])
        if winner:
            self.record['primary_net_rescues']=best[0]
        return winner,masks

    def after_delivery(self,spec,choice,incumbent,history,observed,payloads,profile,executor,timeout,blocked):
        from .autonomous_run import freeze_model, planner_observation
        directory=self.log.run_dir
        self.log.write_json('opening_incumbent.json',incumbent)
        self.record.update(incumbent_sha=incumbent['model_sha256'],incumbent_members=choice['members'],
                           protected_seconds=time.monotonic()-self.start)
        spent=executor.accounting()['attempted_classifier_fits']
        remaining=executor.limit-spent
        self.record.update(fits_at_delivery=spent,remaining_at_delivery=remaining,result='retained')
        self.save()
        self.collect(min(75.,max(0.,(self.deadline-time.monotonic())/4)) if remaining>0 else 0.)
        self.record.update(decision_time_status=self.record['interpretation_status'],decision_seconds=time.monotonic()-self.start)
        base=choice['members']
        confirmations=[f for f in self.folds if f!='primary' and group_observation(observed[f],base,self.metric) is not None
                       and not set(self.folds['primary'][1]).intersection(self.folds[f][1])]
        names=['primary',*confirmations[:1]]
        bases={f:group_observation(observed[f],base,self.metric) for f in names}
        target,masks=self.diagnose(names,bases,observed,base)
        if len(names)<2 or self.record['check_status']=='split_violation':
            self.record['result']='insufficient_matched_evidence'
            self.save()
            return incumbent,choice

        caps=executor.caps
        def affordable(n):
            cost=caps[n]
            missing=sum(n not in observed[f] for f in names)
            return missing*cost['classifier_fits']+cost['refit_classifier_fits']<=remaining
        menu=actions(caps,profile,history,observed,self.metric,list(observed),remaining,blocked)
        local=next((a['members'][0] for a in menu if a['action']=='explore' and affordable(a['members'][0])),None)
        targeted=target is not None and target not in blocked and affordable(target)
        candidate=target if targeted else local
        self.record.update(local_experiment=local,tested_candidate=candidate,changed_experiment=bool(targeted and target!=local),
                           confirmation=names,comparisons=[])
        self.save()  # Freeze the comparison before evaluating any missing fold or refitting.
        if candidate is None:
            self.record['result']='no_affordable_alternative'
            self.save()
            return incumbent,choice
        try:
            scores={}
            cap=caps[candidate]
            for i,f in enumerate(names):
                row=observed[f].get(candidate)
                if row is None:
                    artifact=directory/('opening_probe_'+f+'.joblib')
                    reserve=cap['refit_classifier_fits']+sum(candidate not in observed[g] for g in names[i+1:])*cap['classifier_fits']
                    _,facts=executor.worker(payloads[f]['path'],candidate,artifact,timeout,reserve=reserve)
                    row=planner_observation(joblib.load(artifact),facts,payloads[f],self.metric)
                comparison=compare(f,bases[f],row)
                if f in masks:
                    mask=masks[f]
                    comparison['subset_net_rescues']=int(sum((row['predictions']==row['labels'])[mask])
                                                          -sum((bases[f]['predictions']==row['labels'])[mask]))
                self.record['comparisons'].append(comparison)
                scores[f]=row['score']
                self.save()
            deltas=[r['delta'] for r in self.record['comparisons']]
            if (min(deltas)>0 and np.mean(deltas)>=MEAN_GAIN and
                    (not targeted or self.record['comparisons'][-1]['subset_net_rescues']>0)):
                new_choice=dict(action='stop',members=[candidate],split=None,utility=float(np.mean(list(scores.values()))),
                                evidence=dict(scores=scores),reason='Won both matched training folds under the declared metric.')
                new=freeze_model(self.task,spec,self.metric,new_choice,executor,directory,self.log,timeout,
                                 challenger=True,challenger_prefix='opening_')
                new.update(selection_evidence='task_opening.json',incumbent_model_sha256=incumbent['model_sha256'])
                self.log.write_json('opening_final_selection.json',new)
                self.log.write_json('selection.json',dict(choice=new_choice,scope='matched training folds',evidence='task_opening.json'))
                self.log.write_json('final_selection.json',new)
                self.record.update(result='adopted',winning_members=[candidate])
                return new,new_choice
        except (ValueError,OSError,RuntimeError,AssertionError) as error:
            self.record.update(result='retained_after_probe_failure',probe_issue=str(error)[:300])
            if executor.accounting()['unknown_failed_attempts']:
                raise
        finally:
            self.record['additional_fits']=executor.accounting()['attempted_classifier_fits']-spent
            self.save()
        return incumbent,choice
