from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from superagent import ab_vision
from superagent.ab_catalog import common_catalog
from superagent.metrics import MetricSpec


class FrozenVisualMethodTests(unittest.TestCase):
    def test_resource_and_declared_image_mode_control_availability(self):
        spec = dict(contract_version=1, objective='classification', kind='image',
                    metric=MetricSpec('accuracy',('a','b')).record(), input_schema=dict(modality='image'))
        with patch.object(ab_vision, 'weights_file', return_value=Path('registered')):
            self.assertNotIn(ab_vision.NAME, {r['id'] for r in common_catalog(spec, train_rows=50)})
            spec['input_schema']['image_preprocessing'] = dict(mode='RGB')
            item = next(r for r in common_catalog(spec, train_rows=50) if r['id']==ab_vision.NAME)
            self.assertEqual((item['estimator'],item['expected_classifier_fits'],item['resource']),
                             ('logistic',1,'local_frozen_encoder_cpu'))
        with patch.object(ab_vision, 'weights_file', return_value=None):
            self.assertNotIn(ab_vision.NAME,{r['id'] for r in common_catalog(spec, train_rows=50)})

    def test_invalid_local_weights_are_never_used(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'weights.pth'
            with patch.dict(os.environ, {ab_vision.WEIGHT_ENV:str(path)}):
                self.assertIsNone(ab_vision.weights_file())
                path.write_bytes(b'different resource')
                self.assertIsNone(ab_vision.weights_file())

    def test_decoded_input_requires_an_explicit_byte_image(self):
        self.assertEqual(ab_vision.decoded_images(np.zeros((2,5,6),np.uint8)).shape,(2,5,6,1))
        self.assertEqual(ab_vision.decoded_images(np.zeros((2,5,6,3),np.uint8)).shape,(2,5,6,3))
        for bad in (np.zeros((2,5,6)), np.zeros((2,5,6,5),np.uint8), np.zeros((0,5,6,3),np.uint8)):
            with self.assertRaises(ValueError):
                ab_vision.decoded_images(bad)
