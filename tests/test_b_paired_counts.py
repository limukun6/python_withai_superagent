from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np

from scripts.run_b1_06_mrpc import paired_counts
from scripts.run_b1_ab_acceptance import write


class TestPairedCsvCounts(unittest.TestCase):
    def test_numeric_targets_match_csv_and_the_independent_accuracy_receipt(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / '1').mkdir()
            joblib.dump({'y': np.array([0, 1, 0, 1])}, root / '1/holdout.joblib')
            for name, preds in [('fixed', [0, 1, 1, 1]), ('adaptive', [1, 1, 0, 1])]:
                run = root / ('1_' + name)
                (run / 'pred_0').mkdir(parents=True)
                (run / 'pred_0/rows.csv').write_text('row_id,prediction\n' + ''.join(f'{i},{p}\n' for i, p in enumerate(preds)))
                write(run / 'verification.json', {'rows': 4, 'accuracy': .75})
            result = paired_counts(root, 1, ('fixed', 'adaptive'))
            self.assertEqual([result[key] for key in ('fixed_correct', 'adaptive_correct', 'newly_correct',
                                                        'newly_wrong', 'both_correct', 'both_wrong')], [3, 3, 1, 1, 2, 0])
            write(root / '1_fixed/verification.json', {'rows': 4, 'accuracy': 0})
            with self.assertRaisesRegex(ValueError, 'independent'):
                paired_counts(root, 1, ('fixed', 'adaptive'))
