import unittest

from superagent.explanation_audit import audit_explanation
from superagent.prose_claims import check_prose, classifier_claims, percentage_claims, trial_claims

HISTORY = [
    dict(trial=1, candidate='single_char_C1.0', status='completed', search_accuracy=0.6399,
         train_accuracy=0.836, search_score=0.6399, train_score=0.836, params={'C': 1.0}),
    dict(trial=2, candidate='single_char_C3.0', status='completed', search_accuracy=0.6463,
         train_accuracy=0.84, search_score=0.6463, train_score=0.84, params={'C': 3.0}),
]
CANDIDATE = dict(id='single_char_C3.0', params={'C': 3.0})
FACTS = {'classifier': 'logistic', 'source': 'registered_training_function'}


class ExtractionTests(unittest.TestCase):
    def test_classifier_terms_are_extracted_with_their_compatible_families(self):
        claims = classifier_claims('a character n-gram SVM, not the logistic regression baseline')
        families = [c['claim'] for c in claims]
        self.assertIn('svm', families)
        self.assertIn('logistic', families)

    def test_trial_references_and_percentages(self):
        self.assertEqual([c['claim'] for c in trial_claims('as in trial 2 and Trial 9')], [2, 9])
        self.assertEqual([c['claim'] for c in percentage_claims('64.63% vs 83.00%')], [64.63, 83.0])


class ProseClaimTests(unittest.TestCase):
    def test_a_true_classifier_claim_is_confirmed(self):
        report = check_prose({'reason': 'the logistic regression baseline'}, FACTS, HISTORY)
        self.assertEqual(report['confirmed'], 1)
        self.assertEqual(report['contradicted'], 0)

    def test_a_false_classifier_claim_is_contradicted(self):
        report = check_prose({'reason': 'a character n-gram SVM'}, FACTS, HISTORY)
        self.assertEqual(report['contradicted'], 1)
        self.assertEqual(report['items'][0]['observed'], ['logistic'])

    def test_generic_svm_accepts_either_svm_kernel(self):
        report = check_prose({'reason': 'an rbf SVM'}, {'classifier': 'rbf_svm'}, HISTORY)
        self.assertEqual(report['contradicted'], 0)

    def test_unknown_facts_leave_claims_undecided(self):
        report = check_prose({'reason': 'a character n-gram SVM'}, None, HISTORY)
        self.assertEqual(report['unverifiable'], 1)
        self.assertEqual(report['contradicted'], 0)

    def test_a_classifier_named_in_a_comparison_is_not_a_claim_about_the_candidate(self):
        text = 'depth 4 CatBoost is a stronger probe than only raising the logistic C baseline'
        report = check_prose({'reason': text}, {'classifier': 'catboost'}, HISTORY)
        self.assertEqual(report['contradicted'], 0)
        self.assertEqual(report['informational'], 1)
        referents = {item['claim']: ('other_experiment' if item['in_comparison'] else 'selected_candidate') for item in report['items']
                     if item['kind'] == 'classifier'}
        self.assertEqual(referents['logistic'], 'other_experiment')
        self.assertEqual(referents['catboost'], 'selected_candidate')

    def test_a_claim_without_a_comparison_still_counts(self):
        report = check_prose({'reason': 'the character model here is an SVM'}, {'classifier': 'logistic'},
                             HISTORY)
        self.assertEqual(report['contradicted'], 1)

    def test_a_future_trial_reference_is_contradicted(self):
        report = check_prose({'reason': 'trial 7 showed this'}, FACTS, HISTORY)
        self.assertEqual(report['contradicted'], 1)

    def test_a_measured_percentage_is_confirmed_and_an_invented_one_is_not(self):
        report = check_prose({'reason': 'Trial 2 search accuracy was 64.63%.\n99.99%'}, FACTS, HISTORY)
        self.assertEqual(report['confirmed'], 2)  # existing trial identity plus bound measurement
        self.assertEqual(report['unverifiable'], 1)
        self.assertEqual(report['contradicted'], 0)

    def test_an_unbound_number_match_does_not_certify_a_measurement(self):
        report=check_prose({'reason':'search accuracy 64.63%'},FACTS,HISTORY)
        self.assertEqual(report['confirmed'],0)
        self.assertEqual(report['unverifiable'],1)

    def test_wrong_trial_or_split_cannot_borrow_another_measurement(self):
        for text in ('Trial 1 search accuracy was 64.63%.','Trial 2 search accuracy was 84%.'):
            items=check_prose({'reason':text},FACTS,HISTORY)['items']
            score=next(r for r in items if r['kind']=='percentage')
            self.assertEqual(score['status'],'contradicted')
            self.assertIsNotNone(score['evidence_id'])

    def test_negated_or_future_classifier_is_not_certified(self):
        for text in ('This is not logistic regression.','We might use logistic regression.'):
            report=check_prose({'reason':text},FACTS,HISTORY)
            self.assertEqual(report['confirmed'],0)
            self.assertGreater(report['unverifiable'],0)

    def test_future_trial_plan_is_not_reported_as_a_completed_fact(self):
        report=check_prose({'reason':'Trial 3 will reach 95%.'},FACTS,HISTORY)
        self.assertEqual(report['confirmed'],0)
        self.assertEqual(report['contradicted'],0)

    def test_measurement_in_a_complex_sentence_stays_uncertain(self):
        report=check_prose({'reason':'Trial 2 search accuracy was 64.63% and train accuracy was 84%.'},FACTS,HISTORY)
        self.assertTrue(all(r['status']=='unverifiable' for r in report['items'] if r['kind']=='percentage'))

    def test_empty_fields_produce_no_claims(self):
        self.assertEqual(check_prose({'reason': '', 'review': None}, FACTS, HISTORY)['claims_checked'], 0)


class AuditIntegrationTests(unittest.TestCase):
    def proposal(self, text):
        return {'candidate': 'single_char_C3.0', 'task_kind': 'single_text',
                'reason': text, 'review': 'review text', 'hypothesis': 'h',
                'evidence_trials': [1]}

    def test_audit_reports_prose_counts_and_flags_a_contradiction(self):
        clean = audit_explanation(self.proposal('the logistic regression baseline'), HISTORY, CANDIDATE)
        self.assertFalse(clean['freeform_prose_fully_verified'])
        self.assertIn('prose_claims', clean)
        self.assertGreaterEqual(clean['prose_claims']['claims_checked'], 1)
        self.assertEqual(clean['prose_claims']['contradicted'], 0)

        flagged = audit_explanation(self.proposal('a character n-gram SVM'), HISTORY, CANDIDATE)
        codes = [w['code'] for w in flagged['warnings']]
        self.assertIn('prose_claim_contradiction', codes)
        self.assertEqual(flagged['prose_claims']['unique_contradictions'], ['SVM'])


if __name__ == '__main__':
    unittest.main()
