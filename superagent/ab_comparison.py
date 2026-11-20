"""B-side paired evidence for A's common finalizer. No fit or planner call here."""
import statistics

import numpy as np

PRIMARY = 'primary'
MIN_GAIN = .002


def macro_f1_pair_spread(labels, left, right, domain, groups=None):
    """SD of paired differences; 128 fixed group-bootstrap draws. Used only for triage."""
    y,a,b = (np.asarray(v) for v in (labels,left,right))
    if not (len(y) == len(a) == len(b) and len(y)):
        raise ValueError('Unaligned macro-F1 observations')
    if groups is None:
        group = np.arange(len(y))
    else:
        if len(groups) != len(y):
            raise ValueError('Group identity count differs from the paired rows')
        _,group = np.unique(groups,return_inverse=True)
    count = int(group.max()) + 1
    # Sufficient statistics by group reproduce macro-F1 with the fixed class domain
    # and zero_division=0, without materializing repeated bootstrap rows.
    parts = []
    for pred in (a,b):
        positives,denominators = [],[]
        for label in domain:
            actual,guess = y==label,pred==label
            positives.append(2*np.bincount(group,weights=(actual & guess),minlength=count))
            denominators.append(np.bincount(group,weights=actual.astype(int)+guess.astype(int),minlength=count))
        parts.append((np.stack(positives),np.stack(denominators)))
    rng = np.random.default_rng(902)
    differences = []
    for _ in range(128):
        weights = np.bincount(rng.integers(count,size=count),minlength=count)
        scores = []
        for numerator,denominator in parts:
            den = denominator @ weights
            scores.append(np.divide(numerator @ weights,den,out=np.zeros(len(domain)),where=den>0).mean())
        differences.append(scores[0]-scores[1])
    return float(np.std(differences,ddof=1))


def incumbent_review(measured, records, splits):
    """Replay in trial order. Only hash-aligned completed measurements can promote a challenger."""
    from .metrics import score_of
    ordered = sorted(measured.values(),key=lambda r:r['trial'])
    primary = [r for r in ordered if r.get('split_id',PRIMARY)==PRIMARY and r['trial'] in records]
    if not primary:
        return dict(initial=None,incumbent=None,promotions=[],basis=[])
    initial = min(primary[:3],key=lambda r:(-score_of(r),r['candidate']))
    incumbent = initial['candidate']
    seen, promotions = {}, []
    opening_ends = primary[min(2,len(primary)-1)]['trial']
    for row in ordered:
        name,split = row['candidate'],row.get('split_id',PRIMARY)
        if row['trial'] not in records:
            continue
        seen[name,split] = row
        if row['trial'] <= opening_ends:
            continue
        contenders = {n for n,s in seen if s==PRIMARY and n!=incumbent} if name==incumbent else {name}
        supported = []
        for other in sorted(contenders):
            shared = [s for s in (PRIMARY,*splits) if (incumbent,s) in seen and (other,s) in seen]
            if PRIMARY not in shared or len(shared)<2:
                continue
            differences = [score_of(seen[other,s])-score_of(seen[incumbent,s]) for s in shared]
            gain = statistics.mean(differences)
            if sum(d>0 for d in differences)>=2 and gain>=MIN_GAIN:
                refs = [seen[n,s]['trial'] for s in shared for n in (incumbent,other)]
                supported.append(dict(previous=incumbent,candidate=other,splits=shared,gaps=differences,
                                      mean_gain=gain,evidence_trials=refs))
        if supported:
            decision = min(supported,key=lambda d:(-d['mean_gain'],d['candidate']))
            incumbent = decision['candidate']
            promotions.append(decision)
    basis = promotions[-1]['splits'] if promotions else [PRIMARY]
    return dict(initial=initial['candidate'],incumbent=incumbent,promotions=promotions,basis=basis)
