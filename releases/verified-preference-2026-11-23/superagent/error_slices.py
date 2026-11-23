"""Small descriptive error slices; thresholds and vocabulary fitted on train only."""
from collections import Counter
import re
import numpy as np


def tokens(text):
    return re.findall(r'\b\w+\b', str(text).casefold())


class ErrorSlices:
    def __init__(self, kind, training):
        self.kind = kind
        self.vocabulary = set()
        if kind in ('pair_text', 'single_text'):
            documents = [s for pair in training for s in pair] if kind == 'pair_text' else training
            self.vocabulary = set(w for text in documents for w in tokens(text))
        values = self.features(training)
        self.thresholds = {key: float(np.median(value)) for key, value in values.items()}

    def features(self, inputs):
        if self.kind in ('pair_text', 'single_text'):
            texts = [' '.join(row) for row in inputs] if self.kind == 'pair_text' else inputs
            words = [tokens(s) for s in texts]
            result = {'text_length': np.array([len(w) for w in words]),
                      'unseen_token_fraction': np.array([sum(t not in self.vocabulary for t in w) / max(1, len(w)) for w in words])}
            if self.kind == 'pair_text':
                sets = [(set(tokens(a)), set(tokens(b))) for a, b in inputs]
                result['pair_token_overlap'] = np.array([len(a & b) / max(1, len(a | b)) for a, b in sets])
            return result
        array = getattr(inputs, 'values', inputs)
        if self.kind in ('categorical_tabular', 'mixed_tabular', 'text_tabular'):
            array = np.asarray(array, dtype=str)
            missing = np.isin(np.char.lower(np.char.strip(array)), ['', 'nan', 'none', 'null', 'na', '?'])
            return {'missing_fraction': missing.mean(axis=1)}
        array = np.asarray(array, dtype=float).reshape(len(inputs), -1)
        missing = ~np.isfinite(array)
        clean = np.where(missing, 0., array)
        return {'missing_fraction': missing.mean(axis=1),
                'row_scale': np.mean(np.abs(clean), axis=1),
                'within_row_variation': np.std(clean, axis=1)}

    def describe(self, inputs, labels, predictions, minimum=20):
        truth, pred = np.asarray(labels), np.asarray(predictions)
        if truth.ndim != 1 or pred.shape != truth.shape or len(inputs) != len(truth):
            raise ValueError('Error-slice inputs and predictions must align')
        correct = truth == pred
        overall = float(correct.mean())
        rows = []
        for name, values in self.features(inputs).items():
            threshold = self.thresholds[name]
            for relation, mask in [('low', values <= threshold), ('high', values > threshold)]:
                count = int(mask.sum())
                if count < minimum:
                    continue
                hits = int(correct[mask].sum())
                rows.append({'slice_id': name + '_' + relation, 'feature': name,
                             'threshold_from_training': threshold, 'relation': relation,
                             'rows': count, 'correct': hits, 'accuracy': hits / count,
                             'delta_from_overall_accuracy': hits / count - overall})
        return {'status': 'completed', 'rows': len(truth), 'overall_accuracy': overall,
                'minimum_slice_rows': minimum, 'slices': rows,
                'scope': 'Overlapping descriptive inner-search groups, not causes or significance; thresholds use training inputs only',
                'individual_inputs_or_labels_exposed': False}
