"""Recompute scores, group-bootstrap intervals and fit-receipt counts without training data or weights."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'research/b_title_context_curve_1123'


def read(p):
    return json.loads(p.read_bytes())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def require(ok, why):
    if not ok:
        raise ValueError(why)


def verify_receipts(folder, budget, runner, plan):
    root = DATA/folder
    end = read(root/'completion.json')
    protocol = read(root/'protocol.json')
    result = read(root/'results.json')
    require(all(end[key]==digest(root/(key+'.json')) for key in ('protocol','results')),folder+' seal')
    require(protocol['runner']==digest(ROOT/'scripts'/runner) and protocol['plan']==digest(ROOT/'docs/batches'/plan),
            folder+' code and plan')
    require(end['fits']==result['physical_fits']==budget and
            end['model_api_calls']==result['model_api_calls']==result['new_rmb']==0 and
            not result['course_validation_or_test'],folder+' rows or budget')
    require(protocol.get('fit_cap')==budget and protocol['model_api_cap']==0, folder+' authorized budget')
    fitfiles = list(root.rglob('fit_completed.json'))
    require(len(fitfiles)==budget==len(list(root.rglob('fit_started.json'))),folder+' fit files')
    for f in fitfiles:
        fit = read(f)
        require(fit['fits']==1 and fit['start']==digest(f.with_name('fit_started.json')),folder+' physical fit')
    if folder=='screen':
        for seed, export in end['exports'].items():
            require(digest(root/seed/'exports.json')==export and
                    read(root/seed/'exports.json')['fit']==digest(root/seed/'fit_completed.json'), 'Screen export')
    else:
        require(digest(root/'exports.json')==end['exports'],folder+' export')
        if folder=='confirm':
            require(read(root/'exports.json')['fit']==digest(root/'challenger/fit_completed.json'),'Confirmation fit')
        elif folder=='curve':
            for key, exp in read(root/'exports.json').items():
                for name, sealed in exp['fits'].items():
                    require(sealed==digest(root/key/name/'fit_completed.json'), 'Curve fit')
    return result


def compare(table, panel, old, new, record, expected):
    groups = [items for name,items in table['groups'] if panel is None or panel==name]
    ia, ib = (table['arms'].index(arm) for arm in (old,new))
    counts = []
    for items in groups:
        va = np.array(items)
        a, b = (va >> ia) & 1, (va >> ib) & 1
        counts.append([len(items), a.sum(), b.sum(), ((b==1)&(a==0)).sum(), ((b==0)&(a==1)).sum()])
    g = np.asarray(counts)
    n, first, second, gained, lost = g.sum(axis=0)
    require((first,second)==expected and (gained,lost)==(record['gained'],record['lost']), 'Incorrect paired counts')
    delta = record.get('delta',record)
    require(delta['groups']==len(g) and delta['net_correct']==second-first, 'Incorrect group delta')
    np.testing.assert_allclose(delta['net_pp'],100*(second-first)/n,rtol=0,atol=1e-10)
    rng = np.random.default_rng(60423)
    totals, net = g[:,0], g[:,2]-g[:,1]
    draws = []
    for _ in range(100):
        pick = rng.integers(len(g), size=(50,len(g)))
        draws.extend(100*net[pick].sum(axis=1)/totals[pick].sum(axis=1))
    np.testing.assert_allclose(np.quantile(draws,[.025,.975]),delta['group_bootstrap_95_pp'], rtol=0, atol=1e-10)
    return int(n)


def main():
    title_plan='B_paired_context_1123_zh.md'
    screen=verify_receipts('screen',2,'run_b_paired_context.py',title_plan)
    conf=verify_receipts('confirm',1,'run_b_paired_context_confirmation.py',title_plan)
    curve=verify_receipts('curve',9,'run_b_grouped_learning_curve.py','B_grouped_learning_curve_1123_zh.md')
    vote=verify_receipts('subset',0,'run_b_training_subset_vote.py','B_training_subset_vote_1123_zh.md')
    sp=read(DATA/'screen/protocol.json')
    cp=read(DATA/'confirm/protocol.json')
    lp=read(DATA/'curve/protocol.json')
    require(sp['representation']==digest(ROOT/'scripts/b_contextual_pair_features.py') and
            cp['fit_function']==sp['runner'] and cp['screening']==digest(DATA/'screen/completion.json') and
            cp['partitions']==lp['partitions'] and cp['control']==lp['control'], 'Unmatched code or partitions')
    require(cp['frozen_vectors']==read(DATA/'screen/encoding.json')['sha256'] and
            sp['training']==cp['training']==lp['input'], 'Different input or cached representation')
    data=read(DATA/'grouped_correctness.json')
    for seed,r in screen['panels'].items():
        require(compare(data[seed],None,'E5','context',r,(r['baseline'],r['context']))==r['rows'],seed)
    prim=[r['context']-r['baseline'] for r in screen['panels'].values()]
    require(screen['eligible_for_confirmation']==(min(prim)>=0 and max(prim)>=5), 'Primary gate')
    for key,r in conf['panels'].items():
        require(compare(data['confirmation'],None if key=='combined' else key,'E5','context',r,
                        (r['baseline'],r['context']))==r['rows'],key)
    diffs=[conf['panels'][k]['context']-conf['panels'][k]['baseline'] for k in ('A','B')]
    require(conf['eligible_for_stronger_control']==(min(diffs)>=0 and sum(diffs)>=8), 'Confirmation gate')
    for transition,panels in curve['transitions'].items():
        order,low,high=transition.rsplit('-',2)
        a,b=order+'-'+low,'full' if high=='100' else order+'-'+high
        for panel,r in panels.items():
            require(compare(data['confirmation'],None if panel=='combined' else panel,a,b,r,
                            (curve['correct'][a][panel],curve['correct'][b][panel]))==curve['rows'][panel],transition)
    require(curve['final_step_positive_in_both']==all(curve['correct']['full']['combined']>
            curve['correct'][k+'-75']['combined'] for k in ('curve-a','curve-b')), 'Learning-curve conclusion')
    archive=DATA/'subset'
    require(read(archive/'completion.json')['groups']==digest(archive/'grouped_correctness.json') and
            read(archive/'protocol.json')['curve']==digest(DATA/'curve/completion.json') and
            read(archive/'exports.json')['reused_classifier_heads']==6,'Subset vote identity')
    subset=read(archive/'grouped_correctness.json')
    for panel,r in vote['panels'].items():
        require(r['baseline']==curve['correct']['full'][panel] and compare(subset,None if panel=='combined' else panel,
                'control','mixture',r,(r['baseline'],r['mixture']))==r['rows'],'Wrong subset mixture')
    change=[vote['panels'][k]['mixture']-vote['panels'][k]['baseline'] for k in ('A','B')]
    require(vote['eligible']==(min(change)>=0 and sum(change)>=8),'Subset vote gate')
    print(json.dumps(dict(context_confirm=diffs, curve=curve['correct'], subset_vote=change,
                          physical_fits=12, model_api_calls=0)))


if __name__=='__main__':
    main()
