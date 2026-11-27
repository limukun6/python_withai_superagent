"""Learn shrunk feature-block scales using only the examples in a fitting partition."""
import json
import os
from functools import partial
from pathlib import Path

import numpy as np
from scipy.sparse import hstack

from .paired_encoder import PairedEncoderFeatures


def scales(blocks, mode, labels=None, groups=None):
    count=len(blocks)
    if mode=="energy":
        energy=np.array([float(v.multiply(v).sum())/v.shape[0] for v in blocks])
        signal=np.divide(1.,energy,out=np.zeros_like(energy),where=energy>1e-12)
    elif mode=="aligned":
        y=np.asarray(labels)
        groups=np.asarray(groups)
        if (y.ndim!=1 or groups.shape!=y.shape or not np.isin(y,[0,1]).all()
                or any(v.shape[0]!=len(y) for v in blocks)):
            raise ValueError("Supervised block scaling requires aligned binary preference data")
        rng=np.random.default_rng(2369)
        units=rng.permutation(np.unique(groups))
        if len(units)<2:
            return np.ones(count),np.zeros(count)
        left=np.flatnonzero(np.isin(groups,units[:len(units)//2]))
        right=np.flatnonzero(~np.isin(groups,units[:len(units)//2]))
        sign=2*y.astype(float)-1
        first=rng.choice(left,size=4096)
        second=rng.choice(right,size=4096)
        evidence=[]
        for matrix in blocks:
            mean_l=np.asarray(matrix[left].T@sign[left]).ravel()/len(left)
            mean_r=np.asarray(matrix[right].T@sign[right]).ravel()/len(right)
            squares=0.
            for start in range(0,len(first),256):
                chunk=slice(start,start+256)
                dots=np.asarray(matrix[first[chunk]].multiply(matrix[second[chunk]]).sum(axis=1)).ravel()
                squares+=float(dots@dots)
            rms=np.sqrt(squares/len(first))
            evidence.append(float(mean_l@mean_r)/rms if rms>1e-12 else 0.)
        signal=np.asarray(evidence)
    else:
        raise ValueError("Unknown representation-weighting rule")
    if not np.isfinite(signal).all():
        raise ValueError("Nonfinite feature block statistics")
    positive=np.maximum(signal,0.)
    if positive.sum()<=1e-12:
        return np.ones(count),signal
    kernel=(1.+count*positive/positive.sum())/2.
    return np.sqrt(kernel),signal


class WeightedPairFeatures(PairedEncoderFeatures):
    def __init__(self,variant,head,mode):
        super().__init__(variant,head)
        self.mode=mode

    def fit(self,rows,y=None):
        from .data import TrainingTask
        from .splits import sample_groups
        super().fit(rows,y)
        blocks=self.feature_blocks(rows)
        groups=sample_groups(TrainingTask("pair_text",rows,y),"shared_text") if self.mode=="aligned" else None
        journal=os.getenv("SUPERAGENT_BLOCK_WEIGHT_LOG") if self.mode=="aligned" else None
        if journal:
            with Path(journal).open("a") as f:
                f.write(json.dumps(dict(pid=os.getpid(),event="start"))+"\n")
        self.scales_,self.evidence_=scales(blocks,self.mode,y,groups)
        self.widths_=[v.shape[1] for v in blocks]
        if journal:
            with Path(journal).open("a") as f:
                f.write(json.dumps(dict(pid=os.getpid(),event="complete",scales=self.scales_.tolist()))+"\n")
        return self

    def transform(self,rows):
        blocks=self.feature_blocks(rows)
        if [v.shape[1] for v in blocks]!=self.widths_:
            raise ValueError("Different feature blocks at inference")
        return hstack([v*scale for v,scale in zip(blocks,self.scales_)],format="csr")


def materialize(spec):
    from .models import Candidate
    from .open_methods import digest
    from .paired_encoder import NAMES,candidates,fit_ranker
    if (set(spec)!={"kind","operator","template","mode"} or spec["operator"]!="weight_feature_blocks"
            or spec["kind"]!="pair_text" or spec["mode"] not in ("energy","aligned")
            or NAMES.get(spec["template"],(None,None))[1]!="linear"):
        raise ValueError("Unsupported feature-weight proposal")
    parent=next(item for item in candidates() if item.family==spec["template"])
    variant,head=NAMES[parent.family]
    params=dict(parent.params,parent_template=parent.family,feature_weighting=spec["mode"],
                supervised_weight_stages=int(spec["mode"]=="aligned"))
    fitter=partial(fit_ranker,variant=variant,head=head,feature_type=partial(WeightedPairFeatures,mode=spec["mode"]))
    return Candidate("view_weight_"+digest(spec)[:20],params,fitter)
