"""Exercise the fixed allocation with controlled observations, without training."""
from copy import deepcopy
import unittest
from unittest.mock import patch

from superagent.ab_catalog import capabilities_for, untried_candidates
from superagent.ab_fixed_schedule import fixed_action
from superagent.ab_scheduler import CONFIRMATIONS, PRIMARY
from superagent.scheduler_bridge import validate_action
from tests.test_ab_scheduler import recorded


class FixedScheduleTests(unittest.TestCase):
    def setup_state(self, count=5):
        state = recorded()
        caps = capabilities_for(state['task'], train_rows=256)
        caps = dict(list(caps.items())[:count])
        state.update(history=[], prediction_evidence=[], registered_confirmation_splits=list(CONFIRMATIONS))
        state['budget'].update(fit_limit=11, attempted_classifier_fits=0, completed_classifier_fits=0,
            remaining_fit_allowance=11, reserved_final_refit=1, unknown_failed_attempts=0,
            elapsed_seconds=0, time_limit_seconds=300, llm_request_limit=0, used_llm_requests=0)
        state['available_candidates'] = untried_candidates(caps, [])
        return state, caps

    def test_full_allocation_and_shortlist_frozen(self):
        state,caps = self.setup_state()
        allowed = {s:dict(id=s) for s in (PRIMARY,*CONFIRMATIONS)}
        # Only evidence verification is mocked; the real gate validates every action.
        with patch('superagent.ab_fixed_schedule.selection_evidence'), patch('superagent.ab_fixed_schedule.next_action') as picker:
            for name in list(caps)[:4]:
                picker.return_value=dict(action='explore',candidate_id=name,split_id=PRIMARY,
                    estimated_classifier_fits=1,reason='shared picker',evidence_trials=[])
                action=fixed_action(state,capabilities=caps,trusted_partitions={PRIMARY:{}})
                self.assertEqual(picker.call_args.args[0]['registered_confirmation_splits'],[])
                validate_action(state,action,caps,allowed)
                self.observe(state,caps,action,0.5+0.01*len(state['history']))
            shortlist=[r['candidate'] for r in reversed(state['history'][1:])]
            pairs=[]
            for i in range(6):
                action=fixed_action(state,capabilities=caps,trusted_partitions={})
                validate_action(state,action,caps,allowed)
                pairs.append((action['candidate_id'],action['split_id']))
                # Huge alternating confirmation values cannot change the primary shortlist.
                self.observe(state,caps,action,float(i%2))
            self.assertEqual(pairs,[(n,s) for s in CONFIRMATIONS for n in shortlist])
            action=fixed_action(state,capabilities=caps,trusted_partitions={})
            validate_action(state,action,caps,allowed)
            self.assertEqual(action['action'],'stop')
            self.assertEqual(state['budget']['remaining_fit_allowance'],1)

    def observe(self,state,caps,action,score):
        state['history'].append(dict(trial=len(state['history'])+1,candidate=action['candidate_id'],
            params=caps[action['candidate_id']]['params'],split_id=action['split_id'],status='completed',
            primary_metric='accuracy',search_score=score,search_accuracy=score))
        state['available_candidates']=untried_candidates(caps,state['history'])
        state['budget']['attempted_classifier_fits']+=1
        state['budget']['completed_classifier_fits']+=1
        state['budget']['remaining_fit_allowance']-=1

    def test_small_catalog_does_not_invent_candidates(self):
        state,caps=self.setup_state(1)
        name=next(iter(caps))
        self.observe(state,caps,dict(candidate_id=name,split_id=PRIMARY),.7)
        with patch('superagent.ab_fixed_schedule.selection_evidence'):
            for split in CONFIRMATIONS:
                action=fixed_action(state,capabilities=caps,trusted_partitions={})
                self.assertEqual((action['candidate_id'],action['split_id']),(name,split))
                self.observe(state,caps,action,.7)
            self.assertEqual(fixed_action(state,capabilities=caps,trusted_partitions={})['action'],'stop')
        self.assertEqual(state['budget']['attempted_classifier_fits'],3)

    def test_reject_unknown_or_insufficient_replication_allowance(self):
        original,caps=self.setup_state()
        for name in list(caps)[:4]:
            self.observe(original,caps,dict(candidate_id=name,split_id=PRIMARY),.6)
        for changes in (dict(unknown_failed_attempts=1),dict(remaining_fit_allowance=6),dict(time_limit_seconds=0)):
            state=deepcopy(original);state['budget'].update(changes)
            with patch('superagent.ab_fixed_schedule.selection_evidence'),self.assertRaises(ValueError):
                fixed_action(state,capabilities=caps,trusted_partitions={})

    def test_corrupt_evidence_and_failed_results_stop(self):
        state,caps=self.setup_state()
        self.observe(state,caps,dict(candidate_id=next(iter(caps)),split_id=PRIMARY),.6)
        with patch('superagent.ab_fixed_schedule.selection_evidence',side_effect=ValueError('corrupt evidence')):
            with self.assertRaisesRegex(ValueError,'corrupt evidence'):
                fixed_action(state,capabilities=caps,trusted_partitions={})
        state['history'][0]['status']='failed'
        with self.assertRaisesRegex(ValueError,'incomplete execution'):
            fixed_action(state,capabilities=caps,trusted_partitions={})


if __name__=='__main__':unittest.main()
