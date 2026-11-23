"""Recompute paired counts and audit matched candidate/partition, model and physical-fit records."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def read(f):
    return json.loads(f.read_bytes())


def sha(f):
    return hashlib.sha256(f.read_bytes()).hexdigest()


def check(v,why):
    if not v:
        raise ValueError(why)


def verify(key):
    p=ROOT/'research/b_preference_policy_controls_1123'/key
    end,plan,r,dec=[read(p/(k+'.json')) for k in ('completion','protocol','results','frozen_decisions')]
    for k, file in [('protocol','protocol'),('results','results'),('decisions','frozen_decisions'),('exports','exports')]:
        check(end[k]==sha(p/(file+'.json')),key+' '+k)
    check(plan['plan']==sha(ROOT/'docs/batches/B_preference_policy_controls_1123_zh.md')
          and plan['primary_fit_limit']==11 and end['model_api_calls']==r['model_api_calls']==0,key+' plan or budget')
    a=dec['arms'];parts=dec['partitions']
    check({v['fit_limit'] for n,v in a.items() if n in ('fixed','dynamic')}=={11},'Unequal main budgets')
    for arm,d in a.items():
        f=p/('dynamic_reused_physical.jsonl' if key=='quotes' and arm=='dynamic' else arm+'_physical.jsonl')
        check(sha(f)==d['physical'] and len(f.read_text().splitlines())==r['arms'][arm]['fits']==d['fits'],arm)
        for s in ('primary','confirmation_1','confirmation_2'):
            check(all(parts[arm][s][col]==parts['dynamic'][s][col] for col in ('row_ids_sha256','label_sha256')),s)
        check(d['actions'][-1]['action']=='stop' and r['arms'][arm]['selected']==d['selected'],arm+' terminal')
    check(sum(a[k]['fits'] for k in plan['registered_order'])==r['actual_new_fits']==end['new_fits'],'New fit total')
    table=read(p/'group_counts.json')
    check(table['columns']==['neither','dynamic_only','reference_only','both'],'Contingency definition differs')
    for other,groups in table['by_reference'].items():
        cols=list(map(sum,zip(*groups)))
        delta=r['paired']['dynamic_minus_'+other]
        check(len(groups)==r['groups'] and sum(cols)==r['rows']
              and (cols[1],cols[2])==(delta['gained'],delta['lost'])
              and cols[1]+cols[3]==r['arms']['dynamic']['correct'] and cols[2]+cols[3]==r['arms'][other]['correct']
              and cols[1]-cols[2]==delta['delta']['net_correct'],key+' grouped difference')
    return dict(correct={k:v['correct'] for k,v in r['arms'].items()},new_fits=end['new_fits'],model_api_calls=0)


if __name__=='__main__':
    print(json.dumps({k:verify(k) for k in ('headlines','quotes')}))
