"""Choose early exits on development feedback and physically route only the necessary examples."""
import numpy as np

from superagent.ab_complement import canonical_probabilities

THRESHOLDS = (0., .1, .2, .3, .4, 1.)


def positive_labels(probability, tie_left):
    return ((probability > .5) | ((probability == .5) & tie_left)).astype(int)


def decision_counts(fold, first, second, cutoff):
    p = fold['scores'][first]
    q = fold['scores'][second]
    skip = np.abs(p-.5) >= cutoff
    dual = (p+q)/2
    predicted = positive_labels(np.where(skip,p,dual), fold['tie_left'])
    return dict(threshold=cutoff, rows=len(p), skips=int(skip.sum()), correct=int((predicted==fold['labels']).sum()),
                full_correct=int((positive_labels(dual,fold['tie_left'])==fold['labels']).sum()))


def choose(primary, fresh):
    names = sorted(primary['scores'])
    if len(names)!=2 or set(fresh['scores'])!=set(names) or not len(fresh['labels']):
        raise ValueError('Two matched candidates with fresh confirmation rows are required')
    first_correct = {n:int((positive_labels(primary['scores'][n],primary['tie_left'])==primary['labels']).sum())
                     for n in names}
    first = max(names,key=first_correct.get)
    second = next(n for n in names if n!=first)
    menu = [decision_counts(primary,first,second,t) for t in THRESHOLDS]
    best = max((r for r in menu if r['correct']>=r['full_correct']),key=lambda r:(r['skips'],r['threshold']))
    conf = decision_counts(fresh,first,second,best['threshold'])
    approved = conf['correct']>=conf['full_correct'] and conf['skips']>=.1*conf['rows']
    return dict(first=first,second=second,first_correct=first_correct,primary=menu,proposed=best['threshold'],
                confirmation=conf,adopted=approved,threshold=best['threshold'] if approved else 1.)


def evaluate(model, rows):
    if not rows:
        return np.array([],dtype=float),dict(rows=0,input_tokens=0)
    usage = model.steps[0][1].usage_
    start = usage['input_tokens']
    prob = canonical_probabilities(model,rows)[1][:,1]
    return prob,dict(rows=len(rows),input_tokens=usage['input_tokens']-start)


def predict(models, rows, decision, threshold):
    first,stage_one = evaluate(models[decision['first']],rows)
    needed = np.abs(first-.5) < threshold
    second,stage_two = evaluate(models[decision['second']],[row for row,flag in zip(rows,needed) if flag])
    win = first.copy()
    win[needed] = (win[needed]+second)/2
    tie = np.array([l<r for l,r in rows],dtype=bool)
    return (positive_labels(win,tie),positive_labels(first,tie),needed,
            dict(first=stage_one,second=stage_two,skips=int((~needed).sum())))
