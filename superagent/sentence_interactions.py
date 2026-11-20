"""Antisymmetric nonlinear features for binary preferences between two texts."""
import numpy as np

from .minilm_features import MiniLMTextClassifier


def interaction_features(left,right,projections,opponent):
    first,second=projections
    a,b=left@first,right@first
    c,d=left@second,right@second
    extra=(a*d-b*c) if opponent else (a*c-b*d)
    return np.column_stack((left-right,extra/np.sqrt(2*first.shape[1]))).astype(np.float32)


class NonlinearSentenceClassifier(MiniLMTextClassifier):
    def __init__(self,runtime,params):
        super().__init__('pair_text',runtime,False)
        self.opponent=params['opponent_context']
        rng=np.random.default_rng(params['projection_seed'])
        self.projections=tuple(rng.normal(size=(384,params['rank'])).astype(np.float32) for _ in range(2))

    def dense_features(self,rows):
        if self.encoder is None:
            self.encoder=self.encoder_class(self.runtime)
        vectors=self.encoder.encode([s for pair in rows for s in pair])
        return interaction_features(vectors[::2],vectors[1::2],self.projections,self.opponent)
