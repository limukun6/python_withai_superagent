"""Equal-weight soft vote with the binary preference tie rule used by the paired experiment."""
import numpy as np

from superagent.ab_preference_output import CanonicalPreferencePipeline
from superagent.models import SoftVoteModel


class PreferenceSoftVote(SoftVoteModel):
    def predict(self, inputs):
        p=self.predict_proba(inputs)[:,list(self.classes_).index(1)]
        first=np.array([left<right for left,right in inputs],dtype=bool)
        return ((p>.5)|((p==.5)&first)).astype(self.classes_.dtype)


def canonical_frozen(model):
    return CanonicalPreferencePipeline(list(model.steps))
