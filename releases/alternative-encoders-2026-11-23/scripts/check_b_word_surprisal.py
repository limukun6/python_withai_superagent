"""Measure a fixed, training-text-only surprisal heuristic on four internal search partitions."""
import argparse
from collections import Counter
from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
import re
import sys
from unittest.mock import patch

import joblib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_local_component_pair import check,read,save
from superagent import ab_run
from superagent.ab_complement import canonical_probabilities
from superagent.ab_methods import TEXT_DEEP
from superagent.prediction_evidence import fingerprint
from superagent.splits import normalized_text

TOKENS=re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)?|\d+|[^\w\s]",re.UNICODE)
UNKNOWN='<UNK>'
START='<BOS>'


def tokenize(text):
    return tuple(TOKENS.findall(text.casefold()))


def corpus_scores(titles):
    texts=sorted(set(titles))
    freq=Counter(t for s in texts for t in tokenize(s))
    vocab={k for k,v in freq.items() if v>=2}
    unigram=Counter()
    contexts=Counter()
    bigram=Counter()
    for s in texts:
        previous=START
        for token in tokenize(s):
            token=token if token in vocab else UNKNOWN
            unigram[token]+=1
            contexts[previous]+=1
            bigram[previous,token]+=1
            previous=token
    denominator=sum(unigram.values())+len(vocab)+1
    @lru_cache(maxsize=None)
    def measure(text):
        words=tokenize(text)
        if not words:
            return 0.
        previous=START
        total=0.
        for token in words:
            token=token if token in vocab else UNKNOWN
            marginal=(unigram[token]+1)/denominator
            conditional=(bigram[previous,token]+10*marginal)/(contexts[previous]+10)
            total-=math.log(conditional)
            previous=token
        return total/len(words)
    return measure,dict(unique_training_titles=len(texts),vocabulary=len(vocab),training_tokens=sum(unigram.values()),
        source_title_set_sha256=hashlib.sha256(repr(texts).encode()).hexdigest())


def trial(folder,split,additional):
    row=next((r for r in read(folder/'history.json') if r['candidate']==TEXT_DEEP and r['split_id']==split),None)
    if row:
        return folder/f"trial_{row['trial']}.joblib"
    check(additional and split=='confirmation_1' and read(additional/'protocol.json')['missing']==TEXT_DEEP,'Missing anchor')
    return additional/'extra_confirmation.joblib'


def measure_split(folder,split,additional):
    path=folder/(split+'.joblib')
    payload=joblib.load(path)
    check(read(folder/(split+'_split.json'))['search_rows_sharing_text']==0,'Training/evaluation title overlap')
    evaluate,report=corpus_scores(t for row in payload['X_train'] for t in row)
    held=payload['X_search']
    train_text={normalized_text(t) for row in payload['X_train'] for t in row}
    check(not any(normalized_text(t) in train_text for row in held for t in row),'A held title entered the count corpus')
    delta=np.asarray([evaluate(left)-evaluate(right) for left,right in held])
    order=np.array([a<b for a,b in held],dtype=bool)
    heuristic=((delta>0)|((delta==0)&order)).astype(int)
    source=trial(folder,split,additional)
    artifact=joblib.load(source)
    rec=artifact['prediction_record']
    y=payload['y_search']
    check(rec['evaluation_payload_sha256']==ab_run.sha(path) and rec['label_sha256']==fingerprint(y)
          and artifact['metrics']['resolved_configuration']['id']==TEXT_DEEP,'Wrong anchor or partition')
    hard=np.asarray(artifact['search_predictions'])
    check(fingerprint(hard)==rec['prediction_sha256'],'Bound predictions changed')
    positive=canonical_probabilities(artifact['model'],held)[1][:,1]
    eligible=(np.abs(positive-.5)<=.05)&(np.abs(delta)>=.5)
    fixed=np.where(eligible,heuristic,hard)
    gain=int(((fixed==y)&(hard!=y)).sum())
    lost=int(((fixed!=y)&(hard==y)).sum())
    return dict(**report,rows=len(y),heuristic_correct=int((heuristic==y).sum()),native_correct=int((hard==y).sum()),
        eligible_rows=int(eligible.sum()),changed=int((fixed!=hard).sum()),updated_correct=int((fixed==y).sum()),
        gained=gain,lost=lost,net=gain-lost,held_row_fingerprint=read(folder/(split+'_split.json'))['row_ids_sha256'],
        checkpoint_sha256=ab_run.sha(source))


def run(old,native,ties,out):
    results={}
    for label,folder,additional in [('8123',old/'deep-e5',None),('18643',native/'autonomous',ties)]:
        for split in ('primary','confirmation_1'):
            results[label+'_'+split]=measure_split(folder,split,additional)
    n=sum(v['rows'] for v in results.values())
    net=sum(v['net'] for v in results.values())
    wins=sum(v['net']>0 for v in results.values())
    result=dict(partitions=results,pooled_rows=n,pooled_net=net,pooled_change_pp=100*net/n,partition_wins=wins,
        next_stage_evidence=wins>=3 and net/n>=.002,supervised_fits=0,model_api=0,external_holdout_used=False,
        plan=ab_run.sha(ROOT/'docs/batches/B_word_surprisal_1123_zh.md'))
    save(out,result)
    print({k:(v['heuristic_correct'],v['rows'],v['eligible_rows'],v['net']) for k,v in results.items()},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('workspace','old','native','ties','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    with patch.dict(os.environ,offline_environment(a.workspace.resolve())):
        run(*(getattr(a,n).resolve() for n in ('old','native','ties','output')))
