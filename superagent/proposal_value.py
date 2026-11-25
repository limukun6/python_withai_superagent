"""Training-only evidence, a shared space of executable contrasts and a once-only final comparison."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import math
import time

import joblib
import numpy as np

from .conditional_review import MEAN_GAIN, group_observation
from .data import TrainingTask
from .experiment_policy import actions, identity
from .models import SoftVoteModel, predict_with_model, probability_with_model
from .open_methods import VERSION, capability
from .splits import grouped_partition, take


MODES=('evidence-control','evidence-local','proposal-flash')
PROMPT=(
    'Propose one cheap experiment to explain shared errors from actual, numbered training trials. The ordinary '
    'search continues independently. All listed alternatives are executable. Return concise JSON; do not invent '
    'trial numbers or option indices. If the local experiment is sufficient use {"decision":"keep","reason":"..."}. '
    'Otherwise use {"decision":"experiment","option":0,"cited_trials":[1,2],"hypothesis":"...","refute_if":"...",'
    '"min_rescues":1,"min_gain":0.0}. Cite both reference trials, use an actual zero-based option index and give '
    'a specific hypothesis and a falsifying observation. Rescues counts previously shared errors the new candidate '
    'corrects. min_gain is the required primary metric difference relative to the reference leader in score units. '
    'Cite measured evidence and link the proposed representation or estimator to the hypothesis. The normal '
    'two-fold confirmation and final independent comparison still apply. You can decline. No external scores.'
)


def split_search(task,seed,partition):
    try:
        pool,sealed,info=grouped_partition(task,seed+103,'shared_text')
        inner=TrainingTask(task.kind,take(task.X_train,pool),task.y_train[pool],task.spec)
        regular=partition(inner,seed)
        folds={f:(pool[a],pool[b],r) for f,(a,b,r) in regular.items()}
    except ValueError as error:
        return task,partition(task,seed),np.array([],dtype=int),dict(status='unavailable',reason=str(error)[:240])
    if set(pool)&set(sealed) or any(set(sealed)&(set(a)|set(b)) for a,b,_ in folds.values()):
        raise ValueError('Confirmation rows leaked into search')
    return inner,folds,sealed,dict(status='sealed',pool=pool.tolist(),sealed=sealed.tolist(),protocol=info)


def contrasts(entries, bases):
    """Perturb one tunable setting of a runtime-discovered pipeline, identically in all arms."""
    result={}
    for base in bases:
        if base not in entries:
            continue
        template=entries[base]
        for key in ('C','min_samples_leaf','alpha','n_neighbors','max_depth'):
            if key not in template['kwargs']:
                continue
            old=template['kwargs'][key]
            for factor in (.5,2.):
                value=max(1,round(old*factor)) if type(old) is int else round(old*factor,8)
                if value==old:
                    continue
                spec=dict(template,kwargs=dict(template['kwargs'],**{key:value}))
                try:
                    cap=capability(spec)
                except (ValueError,TypeError):
                    continue
                result[cap['id']]=spec
            break
    return result


def fragile(gains,rows):
    if len(gains)<2:
        return True
    benefits=np.maximum(np.asarray(gains)*rows,0.)
    leave_one=[np.average(np.delete(gains,i),weights=np.delete(rows,i)) for i in range(len(gains))]
    return min(leave_one)<MEAN_GAIN or max(benefits)>2*sum(benefits)/3


def checked_reply(reply,packet):
    if not isinstance(reply,dict):
        raise ValueError('Proposal must be an object')
    if reply.get('decision')=='keep' and isinstance(reply.get('reason'),str) and reply['reason'].strip():
        return dict(decision='keep',reason=reply['reason'][:750])
    idx=reply.get('option')
    cites=reply.get('cited_trials')
    rescued,gain=reply.get('min_rescues'),reply.get('min_gain')
    if not (reply.get('decision')=='experiment' and type(idx) is int and 0<=idx<len(packet['options'])
            and isinstance(cites,list) and all(type(x) is int for x in cites) and sorted(cites)==sorted(packet['reference_trials'])
            and type(rescued) is int and 1<=rescued<=packet['shared_errors']
            and type(gain) in (int,float) and math.isfinite(gain) and 0<=gain<=1-packet['reference_score']+1e-12
            and all(isinstance(reply.get(k),str) and reply[k].strip() for k in ('hypothesis','refute_if'))):
        raise ValueError('Invalid or ungrounded experimental contract')
    return dict(decision='experiment',candidate=packet['options'][idx]['name'],option=idx,cited_trials=cites,
                min_rescues=rescued,min_gain=gain,hypothesis=reply['hypothesis'][:900],refute_if=reply['refute_if'][:750])


class ProposalStudy:
    def __init__(self,mode,directory,log,flash,start,wall):
        self.mode,self.directory,self.log,self.flash=mode,directory,log,flash
        self.start,self.deadline=start,start+wall
        self.record=dict(mode=mode,triggered=False,reply_status='not_requested',seal_opened=False)
        self.snapshots=[]
        self.future=self.threads=self.packet=None
        self.extra={}
        self.snapshot_caps={}
        self.tail_completed=0

    def save(self):
        self.log.write_json('experiment_proposal.json',self.record)

    def snapshot(self,selection,completed,final=False):
        self.snapshots.append(dict(elapsed=time.monotonic()-self.start,completed=completed,final=final,selection=selection))
        self.log.write_json('proposal_checkpoints.json',self.snapshots)

    def launch(self,choice,menu,generator,caps,observed,history,profile,metric,remaining):
        if self.packet is not None or choice['action']=='stop' or remaining<4 or self.deadline-time.monotonic()<20:
            return
        bank=observed['primary']
        if len(bank)<2:
            return
        top=sorted(bank,key=lambda n:(-bank[n]['score'],n))[:2]
        left,right=(bank[n] for n in top)
        if left['row_key']!=right['row_key']:
            return
        shared=(left['labels']!=left['predictions']) & (right['labels']!=right['predictions'])
        count=int(sum(shared))
        if count<max(3,math.ceil(.10*sum(left['labels']!=left['predictions']))):
            return
        ordinary=[a['members'][0] for a in menu if a['action']=='explore']
        roots=[*top,*[n for n in ordinary if n in generator.entries][:3]]
        self.extra=contrasts(generator.entries,roots)
        candidates={n:caps[n] for n in ordinary}
        candidates.update({n:capability(spec) for n,spec in self.extra.items() if n not in bank})
        if not candidates:
            return
        # Represent every distinct estimator/representation and every new contrast; no task-name routing.
        seen=set()
        options=[]
        for name,cap in candidates.items():
            if 2*cap['classifier_fits']+cap['refit_classifier_fits']>remaining:
                continue
            key=(cap.get('estimator_class',cap['estimator']),cap['representation'])
            if key in seen and name not in self.extra:
                continue
            seen.add(key)
            params=cap['params'].get('kwargs',{})
            settings={k:v for k,v in params.items() if k in ('C','min_samples_leaf','alpha','n_neighbors','max_depth')}
            options.append(dict(name=name,estimator=key[0],representation=key[1],settings=settings,fit_cost=cap['classifier_fits']))
            self.snapshot_caps[name]=cap
        if not options:
            return
        refs=[next(r for r in history if r['status']=='completed' and r['candidate']==n and r['split_id']=='primary') for n in top]
        self.capture=(top,shared,left['score'])
        self.packet=dict(metric=metric.name,kind=generator.spec['kind'],rows=len(shared),shared_errors=count,
            shared_errors_by_label=dict(Counter(str(v) for v in left['labels'][shared])),
            reference_trials=[r['trial'] for r in refs],reference_score=left['score'],
            measured=[dict(trial=r['trial'],method=r['candidate'],score=r['search_score'],train_score=r['train_score'],
                           recall=r['class_recall']) for r in refs],minority_fraction=profile['minority_class_fraction'],
            ordinary_next={k:choice[k] for k in ('action','members','split')},local_exploration=ordinary[:1],
            remaining_fits=remaining,options=options)
        self.log.write_json('experiment_proposal_input.json',self.packet)
        self.record.update(triggered=True,trigger_after_trials=len(history),input_digest=identity(self.packet),
                           shared_errors=count,options=len(options),new_configurations=len(self.extra))
        if self.mode=='proposal-flash' and self.flash is not None:
            self.flash.set_deadline(min(self.deadline-5,time.monotonic()+90))
            self.threads=ThreadPoolExecutor(max_workers=1)
            self.future=self.threads.submit(self.flash.ask,dict(instruction=PROMPT,evidence=self.packet),
                                           self.log,'experiment_proposal',cost_ceiling=.12)
            self.record['reply_status']='pending'
        elif self.mode=='proposal-flash':
            self.record['reply_status']='unavailable'
        self.save()

    def collect(self,wait):
        if self.future is None or self.record['reply_status']!='pending':
            return self.record.get('plan')
        try:
            reply=self.future.result(timeout=max(0.,wait))
        except FutureTimeout:
            if self.future.done():
                self.record.update(reply_status='invalid',reply_error='Request timeout')
            return None
        except (ValueError,TypeError,OSError,RuntimeError) as error:
            self.record.update(reply_status='invalid',reply_error=str(error)[:300])
            self.save()
            return None
        try:
            plan=checked_reply(reply,self.packet)
            self.record.update(plan=plan,reply_status='valid')
        except (ValueError,TypeError,KeyError) as error:
            self.record.update(reply_status='invalid',reply_error=str(error)[:300])
        self.save()
        return self.record.get('plan')

    def finish(self):
        if self.threads is not None:
            self.threads.shutdown(wait=True)
            self.collect(0.)
        self.record['closed_seconds']=time.monotonic()-self.start
        self.save()

    def tail(self,task,spec,metric,core,checkpoint,history,observed,payloads,sealed,generator,executor,profile,timeout,blocked):
        from .autonomous_run import planner_observation
        caps=dict(self.snapshot_caps,**executor.caps)
        caps.update({n:capability(s) for n,s in self.extra.items()})
        executor.caps=caps
        if self.extra:
            document=dict(schema_version=VERSION,kind=task.kind,entries={**generator.entries,**self.extra})
            self.log.write_json('candidate_registry.json',document)
        remaining=executor.limit-executor.accounting()['attempted_classifier_fits']
        ranked=actions(caps,profile,history,observed,metric,list(observed),remaining,blocked)
        core_members=core['members']
        backup=checkpoint.get('single_candidate')
        if backup is None:
            self.record.update(guard_action='no_verified_checkpoint',final_members=core_members)
            self.save()
            return core
        reserve=max(sum(caps[n]['refit_classifier_fits'] for n in core_members),caps[backup]['refit_classifier_fits'])
        folds=['primary',*sorted(f for f in core['evidence']['scores'] if f.startswith('confirmation_'))[:1]]
        def affordable(name):
            cap=caps[name]
            missing=sum(name not in observed[f] for f in folds)
            return missing*cap['classifier_fits']+max(reserve,cap['refit_classifier_fits'])<=remaining
        local=next((a['members'][0] for a in ranked if a['action']=='explore' and affordable(a['members'][0])),None)
        plan=self.collect(min(12.,max(0.,(self.deadline-time.monotonic())/5))) if local and len(folds)==2 else self.collect(0.)
        name=local
        use=bool(plan and plan['decision']=='experiment' and len(folds)==2 and plan['candidate'] in caps
                 and plan['candidate'] not in blocked and plan['candidate'] not in core_members and affordable(plan['candidate']))
        if use:
            name=plan['candidate']
        self.record.update(local_alternative=local,tested_candidate=name if len(folds)==2 else None,
                           proposal_used=use,decision_time_status=self.record['reply_status'],
                           core_members=core_members,backup=backup,remaining_at_tail=remaining)
        self.log.write_json('tail_choice.json',dict(candidate=name,local=local,proposal_used=use,plan=plan,folds=folds,
                            ordinary=core_members,reserved_refits=reserve,remaining=remaining))
        self.save()
        selected=core
        files={r['candidate']:self.directory/f"trial_{r['trial']}.joblib" for r in history if r['status']=='completed' and r['split_id']=='primary'}
        if name is not None and len(folds)==2 and name not in core_members:
            comparisons=[]
            try:
                cap=caps[name]
                for i,fold in enumerate(folds):
                    if name not in observed[fold]:
                        path=self.directory/f'proposal_{fold}.joblib'
                        later=sum(name not in observed[f] for f in folds[i+1:])*cap['classifier_fits']
                        _,facts=executor.worker(payloads[fold]['path'],name,path,timeout,
                                      reserve=max(reserve,cap['refit_classifier_fits'])+later)
                        observed[fold][name]=planner_observation(joblib.load(path),facts,payloads[fold],metric)
                        self.tail_completed+=1
                        self.snapshot(checkpoint,sum(r['status']=='completed' for r in history)+self.tail_completed)
                        if fold=='primary':
                            files[name]=path
                    row=observed[fold][name]
                    comparisons.append(dict(fold=fold,challenger=row['score'],core=core['evidence']['scores'][fold],
                                             delta=row['score']-core['evidence']['scores'][fold]))
                    if i==0 and use:
                        _,shared,ref=self.capture
                        rescued=int(sum(shared & (row['predictions']==row['labels'])))
                        passed=rescued>=plan['min_rescues'] and row['score']-ref+1e-12>=plan['min_gain']
                        self.record.update(rescued=rescued,measured_primary_gain=row['score']-ref,hypothesis_passed=bool(passed))
                        if not passed:
                            break
                if len(comparisons)==2 and min(r['delta'] for r in comparisons)>0 and np.mean([r['delta'] for r in comparisons])>=MEAN_GAIN:
                    scores={r['fold']:r['challenger'] for r in comparisons}
                    selected=dict(action='stop',members=[name],split=None,utility=float(np.mean(list(scores.values()))),
                                  evidence=dict(scores=scores,refit_fits=cap['refit_classifier_fits']),reason='Two measured challenger wins.')
            except (ValueError,RuntimeError,OSError,TypeError) as error:
                self.record['tail_error']=str(error)[:300]
                if executor.accounting()['unknown_failed_attempts']:
                    raise
            self.record['screen']=comparisons
        self.record['pre_guard_members']=selected['members']
        if self.mode!='evidence-control' and selected['members']!=[backup]:
            selected=self.guard(task,metric,selected,backup,files,observed,sealed)
        self.record['final_members']=selected['members']
        self.save()
        return selected

    def guard(self,task,metric,selected,backup,files,observed,sealed):
        members=selected['members']
        diffs,rows=[],[]
        for bank in observed.values():
            candidate=group_observation(bank,members,metric)
            if backup in bank and candidate is not None:
                base=bank[backup]
                if candidate['row_key']!=base['row_key']:
                    raise ValueError('Unmatched primary evidence')
                diffs.append(candidate['score']-base['score'])
                rows.append(len(base['labels']))
        self.record.update(fold_gains=diffs,fold_rows=rows)
        if not fragile(diffs,rows):
            self.record['guard_action']='accept_supported_change'
            return selected
        self.log.write_json('guard_choice_before_labels.json',dict(incumbent=backup,challenger=members,fold_gains=diffs,
                            minimum_gain=MEAN_GAIN,rows=len(sealed)))
        scores={fold:bank[backup]['score'] for fold,bank in observed.items() if backup in bank}
        fallback=dict(action='stop',members=[backup],split=None,utility=float(np.mean(list(scores.values()))),
                      evidence={'scores':scores},reason='Retain the verified champion after checking the change.')
        if not len(sealed):
            self.record['guard_action']='keep_unconfirmed'
            return fallback
        try:
            x=take(task.X_train,sealed)
            predictions=[]
            for group in ([backup],members):
                models=[joblib.load(files[n])['model'] for n in group]
                model=models[0] if len(models)==1 else SoftVoteModel(task.kind,models,np.asarray(metric.labels))
                pred=predict_with_model(task.kind,model,x)
                order,probs=probability_with_model(task.kind,model,x) if metric.requires_scores else (None,None)
                predictions.append((pred,probs,order))
            y=task.y_train[sealed]
            self.record['seal_opened']=True
            old,new=[metric.evaluate(y,*p)['score'] for p in predictions]
            self.record.update(sealed_scores=[old,new],guard_action='adopt_confirmed' if new-old>=MEAN_GAIN else 'keep_on_confirmation')
            return selected if new-old>=MEAN_GAIN else fallback
        except (ValueError,RuntimeError,OSError,KeyError,TypeError) as error:
            self.record.update(guard_action='keep_on_error',guard_error=str(error)[:250])
            return fallback
