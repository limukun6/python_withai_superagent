"""Use a planned confirmation to decide which already-trained checkpoint to publish."""
import time

from .autonomous_checkpoint import promote
from .conditional_review import MEAN_GAIN, compare, evaluate_probe, feasible_probes, validate
from .experiment_policy import identity


PUBLISH = 'publish_challenger_checkpoint'
RETAIN = 'keep_current_checkpoint'
PROMPT = (
    'Two training folds disagree. Another measurement is already scheduled by the planner; it will run either way. '
    'Decide whether its result should change the interim prediction model. The full search continues in both cases. '
    'Use only the provided evidence. Output concise JSON. Keep: {"decision":"keep","reason":"..."}. '
    'Alternatively use {"decision":"measure","option":0,"cited_folds":["primary","confirmation_1"],'
    '"if_pass":"publish_challenger_checkpoint","if_fail":"keep_current_checkpoint","minimum_delta":null,'
    '"reason":"...","opportunity_cost":"..."}. Use the actual option number. Null uses the displayed threshold. '
    'The new measured delta must also be strictly positive, and the row-weighted gain over three folds at least '
    'minimum_weighted_gain. You may require a higher feasible delta. Switching changes the interim artifact and '
    'costs no incremental training. Consider sample size, the type of metric, and the consequences of switching.'
)


class ScheduledReview:
    def __init__(self, log, flash, deadline):
        self.log, self.flash, self.deadline = log, flash, deadline
        self.record = dict(status='skipped',reason='no_decisive_scheduled_confirmation',executed=RETAIN)
        self.probe = None

    def save(self):
        self.log.write_json('scheduled_review.json',self.record)

    def before(self, choice, trial, checkpoint, observed, folds, caps, metric, remaining):
        if 'trial' in self.record or choice['action']!='confirm' or checkpoint['model_format']!='verified_trial_v1':
            return
        incumbent=checkpoint['single_candidate']
        try:
            options,_=feasible_probes([incumbent],observed,folds,caps,metric,remaining,scheduled=choice)
        except (KeyError,ValueError,TypeError) as error:
            self.log.event('scheduled_review_evidence_unavailable',trial=trial,reason=str(error)[:360])
            self.record.update(status='skipped_invalid_evidence',reason='training_comparison_unavailable')
            self.save()
            return
        if not options:
            return
        packet=dict(metric=metric.name,incumbent=incumbent,options=options,remaining_fits=remaining,
                    scheduled_method=choice['members'][0],scheduled_fold=choice['split'],incremental_training_fits=0,
                    minimum_weighted_gain=MEAN_GAIN)
        self.record.update(trial=trial,status='awaiting_review',input_digest=identity(packet),incumbent=incumbent,
                           incumbent_sha256=checkpoint['model_sha256'],reason='rank_reversal_before_planned_confirmation')
        self.log.write_json('scheduled_review_input.json',packet)
        self.log.write_json('scheduled_selection_before.json',checkpoint)
        self.save()
        try:
            if self.flash is None:
                raise ValueError('No reviewer configured')
            now=time.monotonic()
            if self.deadline-now<=3:
                raise ValueError('Insufficient wall time for review and remaining search')
            # Leave most of the remaining task wall time for experimentation and delivery.
            self.flash.set_deadline(now+min(120.,(self.deadline-now)/3))
            answer=self.flash.ask(dict(instruction=PROMPT,evidence=packet),self.log,'feedback_reflection',cost_ceiling=.12)
            plan=validate(answer,packet,pass_action=PUBLISH,fail_action=RETAIN)
            self.record.update(plan=plan,status='declined' if plan['decision']=='keep' else 'plan_frozen')
            if plan['decision']=='measure':
                self.probe=options[plan['option']]
        except (ValueError,TypeError,KeyError,RuntimeError,OSError) as error:
            self.record.update(status='retained_after_error',reason=str(error)[:360])
        finally:
            if self.flash is not None:
                self.flash.set_deadline(self.deadline)
        self.save()

    def after(self, trial, checkpoint, observed, task, metric, history, primary, directory, failed=False):
        if trial!=self.record.get('trial'):
            return
        published=checkpoint
        if failed:
            self.record.update(status='scheduled_fit_failed')
        elif self.probe is not None:
            try:
                probe=self.probe
                name,fold=probe['challenger'],probe['third_fold']
                latest=compare(fold,observed[fold][self.record['incumbent']],observed[fold][name])
                weighted,passed=evaluate_probe(probe,latest,self.record['plan']['minimum_delta'])
                self.record.update(third=latest,weighted_delta=weighted,passed=passed,status='measured')
                self.save()
                if passed:
                    row=next(r for r in history if r['candidate']==name and r['split_id']=='primary' and r['status']=='completed')
                    number=row['trial']
                    published=promote(task,metric,checkpoint,number,name,directory/f'trial_{number}.joblib',
                                      primary,observed['primary'][name],self.log,review_evidence=self.record)
                    self.record.update(status='published',executed=PUBLISH)
            except (ValueError,TypeError,KeyError,RuntimeError,OSError,AssertionError,StopIteration) as error:
                self.record.update(status='retained_after_error',reason=str(error)[:360])
        self.log.write_json('scheduled_selection_after.json',published)
        self.record['interim_sha256']=published['model_sha256']
        self.save()

    def finish(self,selection):
        self.record['delivered_sha256']=selection['model_sha256']
        self.save()
