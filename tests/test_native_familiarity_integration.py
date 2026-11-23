import json
import math
from pathlib import Path
import unittest

from superagent.ab_sentence import DEEP_FILES, DEEP_REVISION
from superagent.word_familiarity import WordFamiliarity


class NativeFamiliarityIntegrationTests(unittest.TestCase):
    def test_unseen_and_seen_titles_use_the_expected_reference_counts(self):
        f = WordFamiliarity().fit(['red blue', 'red green', 'blue green', 'red green'])
        self.assertAlmostEqual(f.score('RED'), -math.log(5/13))
        self.assertAlmostEqual(f.score('red blue'), -(math.log(7/24) + math.log(5/22))/2)
        first, reverse = f.differences([('red blue','RED'),('RED','red blue')])
        self.assertEqual(first, -reverse)

    def test_joint_candidates_keep_the_A113R_verified_deep_asset(self):
        root = Path(__file__).resolve().parents[1]
        actual = json.loads((root / 'research/a113R_deep_assets/assets.json').read_text())
        self.assertEqual(DEEP_REVISION, 'a50ef00143b4d5391434df20ae11632588ac25be')
        self.assertEqual(DEEP_FILES, {a['name']:(a['bytes'], a['sha256']) for a in actual})
