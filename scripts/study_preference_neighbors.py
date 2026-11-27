"""Screen a learned vote by analogous training pairs without sharing a title group across roles."""
import argparse
import hashlib
import os
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

from scripts.compare_regrouped_portfolios import lock
from scripts.unified_released_study import ROOT, read, require, sha
from superagent.autonomous_run import planner_observation
from superagent.data import TrainingTask
from superagent.external_assets import activate_resource_directory
from superagent.local_sentence import LocalSentenceFeatures, vectors
from superagent.metrics import MetricSpec
from superagent.paired_encoder import PairedEncoderFeatures, vectors_for
from superagent.prediction_evidence import fingerprint
from superagent.splits import sample_groups


PLAN = ROOT/'research/paired_encoder_search_1127/preference_neighbors_protocol.json'
WEIGHTS = (0., .25, .5, 1.)


def unit(values):
    return values/np.maximum(np.linalg.norm(values, axis=1, keepdims=True),1e-12)


def views(rows, provider):
    # Embed each individual text without labels, or features from other rows.
    unique = list(dict.fromkeys(x for pair in rows for x in pair))
    lookup = {text:i for i,text in enumerate(unique)}
    encoded = provider(unique)
    left, right = (encoded[[lookup[pair[t]] for pair in rows]] for t in (0,1))
    return unit(left-right), unit(left+right)


class AnalogyVote:
    def fit(self, rows, labels, groups, provider):
        self.change, self.context = views(rows, provider)
        self.signed = 2*np.asarray(labels,dtype=float)-1
        require(set(self.signed)=={-1.,1.},'Expected a two-sided preference target')
        self.groups = np.asarray(groups)
        self.k = int(np.clip(round(np.sqrt(len(set(groups)))),3,40))
        # Resolve exact similarities by input content, consistently for swapped sides.
        self.order = np.asarray(sorted(range(len(rows)),key=lambda i:(
            tuple(sorted(rows[i])), int(self.signed[i]*np.sign((rows[i][0]>rows[i][1])-(rows[i][0]<rows[i][1]))))))
        return self

    def predict_probability(self, rows, provider):
        diff, middle = views(rows, provider)
        output = np.full(len(rows), .5, dtype=float)
        for start in range(0, len(rows), 96):
            directional = diff[start:start+96] @ self.change.T
            context = np.maximum(0.,middle[start:start+96] @ self.context.T)
            similarity = np.abs(directional)*context
            for offset, scores in enumerate(similarity):
                # Try the closest rows first and expand only when many share the same training group.
                stable = scores[self.order]
                count = min(len(stable),max(4*self.k,self.k))
                while True:
                    minimum = np.partition(stable,len(stable)-count)[len(stable)-count]
                    eligible = np.flatnonzero(stable>=minimum)
                    closest = self.order[eligible[np.argsort(-stable[eligible],kind='stable')]]
                    selected, used = [],set()
                    for j in closest:
                        if scores[j]<=0:
                            break
                        if self.groups[j] not in used:
                            used.add(self.groups[j])
                            selected.append(j)
                            if len(selected)==self.k:
                                break
                    if len(selected)==self.k or count==len(stable) or minimum<=0:
                        break
                    count = min(2*count,len(stable))
                if selected:
                    weights = np.square(scores[selected]).astype(float)
                    votes = self.signed[selected]*np.sign(directional[offset,selected])
                    margin = np.dot(weights,votes)/(weights.sum()+2*weights.mean())
                    output[start+offset] = (1.+margin)/2.
        return output


def vector_provider(model):
    feature = model.named_steps['features']
    if isinstance(feature,PairedEncoderFeatures):
        return (lambda texts:vectors_for(feature.variant,texts)),feature.encoder_model
    if isinstance(feature,LocalSentenceFeatures):
        return (lambda texts:vectors(feature.spec,texts)),feature.spec['signature']
    raise ValueError('The elected incumbent does not expose a compatible standalone vector provider')


def mixed(anchor, neighbour, weight):
    return (1-weight)*anchor+weight*neighbour


def correct(probabilities, truth):
    return (probabilities>.5)==truth


def select_blends(anchor, neighbour, truth, groups, output, name):
    roles=np.asarray([int.from_bytes(hashlib.sha256(('analogy vote selection v1/'+str(g)).encode()).digest()[:8],'big')%2 for g in groups])
    selected=np.empty(len(truth),dtype=bool)
    actions=[]
    for i in (0,1):
        learn,held=roles!=i,roles==i
        require(learn.any() and held.any(),'Missing disjoint selection or evaluation groups')
        counts=[int(correct(mixed(anchor[learn],neighbour[learn],w),truth[learn]).sum()) for w in WEIGHTS]
        weight=WEIGHTS[int(np.argmax(counts))]
        decision=dict(learning_rows=int(learn.sum()),evaluation_rows=int(held.sum()),counts=counts,weight=weight)
        lock(output/f'{name}_choice_{i}.json',decision)
        selected[held]=correct(mixed(anchor[held],neighbour[held],weight),truth[held])
        actions.append(decision)
    baseline=correct(anchor,truth)
    return dict(anchor=int(baseline.sum()),standalone=int(correct(neighbour,truth).sum()),
                fixed_quarter=int(correct(mixed(anchor,neighbour,.25),truth).sum()),chosen=int(selected.sum()),
                gained=int((selected&~baseline).sum()),lost=int((baseline&~selected).sum()),actions=actions)


def run(root):
    output=root/'preference-neighbors'
    output.mkdir(exist_ok=True)
    resource=root/'regrouped-runs'
    previous=read(resource/'lock.json')
    election=root/'discovered-ordinary/discovered_default/sealed.json'
    member=previous['recipes']['ordinary']['members']
    selected=read(root/'discovered-ordinary/discovered_default/run/agent/decisions.json')[-1]['choice']['members']
    require(len(member)==1 and member==selected,'Different source model than the genuine ordinary election')
    supervised=root/'raw/race/run/agent/refit_data.joblib'
    require(sha(supervised)==previous['training_sha256'],'Training input changed')
    data=joblib.load(supervised)
    x,y=data['X_train'],np.asarray(data['y_train'])
    groups=sample_groups(TrainingTask(data['kind'],x,y),'shared_text')
    activate_resource_directory(root/'discovery-library')
    os.environ['SUPERAGENT_PAIRED_FEATURE_CACHE']=str(root/'feature_bank')
    folds=[]
    for seed in previous['seeds']:
        parts=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=seed).split(groups,y,groups)
        for i,(fit,held) in enumerate(parts,1):
            directory=resource/f'seed_{seed}'/f'fold_{i}'
            payload=directory/'training.joblib'
            artifact=directory/member[0]/'trial.joblib'
            stamp,receipt=read(directory/'partition.json'),read(artifact.parent/'receipt.json')
            require(sha(payload)==stamp['payload_sha256']==receipt['payload_sha256'] and
                    sha(artifact)==receipt['trial_sha256'] and stamp['fit_sha256']==fingerprint(fit)
                    and stamp['held_sha256']==fingerprint(held) and not set(groups[fit])&set(groups[held]),'Fold mismatch')
            folds.append((seed,i,fit,held,payload,artifact,stamp))
    lock(output/'inputs.json',dict(protocol=sha(PLAN),training_sha256=sha(supervised),election=sha(election),
         member=member[0],folds=[dict(seed=s,fold=i,stamp=h,artifact=sha(a)) for s,i,_,_,p,a,h in folds]))
    records=[]
    for seed,i,fit,held,payload,artifact,stamp in folds:
        name=f's{seed}_f{i}'
        trial=joblib.load(artifact)
        packet=joblib.load(payload)
        train=[x[j] for j in fit]
        query=[x[j] for j in held]
        require(train==packet['X_train'] and query==packet['X_search'] and np.array_equal(y[held],packet['y_search']),
                'Mismatch in training/held rows')
        metric=MetricSpec.from_payload(packet)
        obs=planner_observation(trial,trial['metrics'],dict(sha256=sha(payload),labels=y[held]),metric)
        anchor=np.asarray(obs['probabilities'])[:,list(metric.labels).index(1)]
        provider,encoder=vector_provider(trial['model'])
        # A labelled index is one supervised fit, even though the frozen encoder receives no updates.
        vote=AnalogyVote().fit(train,y[fit],groups[fit],provider)
        neighbour=vote.predict_probability(query,provider)
        flip=vote.predict_probability([(b,a) for a,b in query[:21]],provider)
        require(np.allclose(neighbour[:21]+flip,1,rtol=0,atol=1e-9),'Pair swapping changed vote direction')
        lock(output/(name+'_pred.json'),dict(input=sha(output/'inputs.json'),anchor=fingerprint(anchor),
             neighbour=fingerprint(neighbour),encoder=encoder,k=vote.k,supervised_index_fits=1))
        measures=select_blends(anchor,neighbour,y[held],groups[held],output,name)
        records.append(dict(seed=seed,fold=i,rows=len(held),**measures))
        print(seed,i,measures['anchor'],measures['chosen'],[d['weight'] for d in measures['actions']],flush=True)
    totals={}
    for seed in previous['seeds']:
        rows=[r for r in records if r['seed']==seed]
        totals[str(seed)]={k:sum(r[k] for r in rows) for k in ('rows','anchor','standalone','fixed_quarter','chosen','gained','lost')}
    fractions=[(r['chosen']-r['anchor'])/r['rows'] for r in records]
    passed=all((v['chosen']-v['anchor'])/v['rows']>=.002 for v in totals.values()) and sum(f>=0 for f in fractions)>=6 and min(fractions)>=-.005
    lock(output/'results.json',dict(inputs=sha(output/'inputs.json'),totals=totals,folds=records,advance=bool(passed),
         new_supervised_index_fits=len(records),independent_blend_decisions=2*len(records),model_api_requests=0,
         designated_validation_opened=False,formal_test_opened=False,H2_opened=False))
    print(totals,passed,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=lambda s:Path(s).resolve(),required=True)
    with threadpool_limits(limits=3):
        run(parser.parse_args().root)
