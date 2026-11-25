import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
import unittest
from unittest.mock import Mock, patch

from superagent.conditional_review import feasible_probes
from superagent.events import RunLogger
from superagent.scheduled_review import PUBLISH, RETAIN, ScheduledReview
from tests.test_conditional_review import observed, proposal, setup


def action(member='b',fold='confirmation_2'):
    return dict(action='confirm',members=[member],split=fold,evidence={'reserved_refits':1})


class ScheduledReviewTests(unittest.TestCase):
    def test_unmatched_evidence_preserves_planned_fit_and_allows_a_later_verified_review(self):
        with TemporaryDirectory() as tmp:
            folder=Path(tmp)
            banks,folds,caps,metric=setup()
            banks['primary']['b']['row_key']='different-rows'
            client=Mock()
            def advise(*args,**kwargs):
                self.assertEqual(json.loads((folder/'scheduled_review.json').read_text())['status'],'awaiting_review')
                return dict(decision='keep',reason='Little to gain')
            client.ask.side_effect=advise
            review=ScheduledReview(RunLogger(folder),client,time.monotonic()+500)
            checkpoint=dict(model_format='verified_trial_v1',single_candidate='a',model_sha256='baseline')
            review.before(action(),5,checkpoint,banks,folds,caps,metric,6)
            client.ask.assert_not_called()
            self.assertEqual(review.record['status'],'skipped_invalid_evidence')
            banks['primary']['b']['row_key']='primary'
            review.before(action(),6,checkpoint,banks,folds,caps,metric,5)
            self.assertEqual(review.record['status'],'declined')
            client.ask.assert_called_once()

    def test_only_the_measurement_already_selected_can_trigger(self):
        banks,folds,caps,metric=setup()
        def options(choice,remaining=3):
            return feasible_probes(['a'],banks,folds,caps,metric,remaining,scheduled=choice)[0]
        self.assertEqual(options(action())[0]['incumbent_known_third_score'],.7)
        self.assertFalse(options(action('a')))
        self.assertFalse(options(action(fold='confirmation_1')))
        self.assertFalse(options(action(),1))
        del banks['confirmation_2']['a']
        self.assertFalse(options(action()))
        banks['confirmation_2']['b']=observed('confirmation_2',8)
        self.assertEqual(options(action('a'))[0]['challenger_known_third_score'],.8)

    def test_preregistration_binds_two_real_checkpoint_actions_without_training(self):
        keep=dict(decision='keep',reason='Small folds.')
        plan=proposal(if_pass=PUBLISH,if_fail=RETAIN)
        for reply,result,fail,expected in [(plan,9,False,'published'),(plan,6,False,'measured'),
            (keep,9,False,'declined'),(proposal(),9,False,'retained_after_error'),(plan,9,True,'scheduled_fit_failed')]:
            with self.subTest(expected=expected),TemporaryDirectory() as tmp:
                folder=Path(tmp)
                banks,folds,caps,metric=setup()
                client=Mock()
                client.ask.return_value=reply
                review=ScheduledReview(RunLogger(folder),client,time.monotonic()+500)
                selection=dict(model_format='verified_trial_v1',single_candidate='a',model_sha256='baseline')
                review.before(action(fold='confirmation_1'),4,selection,banks,folds,caps,metric,7)
                client.ask.assert_not_called()
                review.before(action(),5,selection,banks,folds,caps,metric,6)
                record=json.loads((folder/'scheduled_review.json').read_text())
                self.assertNotIn('third',record)
                self.assertEqual((folder/'scheduled_selection_after.json').exists(),False)
                review.before(action(),6,selection,banks,folds,caps,metric,5)
                client.ask.assert_called_once()
                banks['confirmation_2']['b']=observed('confirmation_2',result)
                history=[dict(candidate='b',split_id='primary',trial=2,status='completed')]
                with patch('superagent.scheduled_review.promote',return_value={'model_sha256':'challenger'}) as publish:
                    review.after(5,selection,banks,None,metric,history,{},folder,failed=fail)
                self.assertEqual(review.record['status'],expected)
                self.assertEqual(publish.called,expected=='published')
                self.assertEqual(review.record['interim_sha256'],'challenger' if publish.called else 'baseline')
                review.finish(dict(model_sha256='final'))
                self.assertEqual(review.record['delivered_sha256'],'final')
