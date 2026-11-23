"""Recompute the shared-title study's promotion decisions and check its four physical fit receipts."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / 'research/b_shared_title_weighting_1123'


def read(file):
    return json.loads(file.read_bytes())


def sha(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def check(ok, message):
    if not ok:
        raise ValueError(message)


def verify():
    result = {}
    for stage, arms, script, doc in (
        ('screen', ('18643','40531'), 'run_b_shared_title_weighting','B_shared_title_weighting_1123_zh'),
        ('confirm', ('regular','weighted'), 'run_b_shared_title_confirmation','B_shared_title_confirmation_1123_zh')):
        folder = PACK / stage
        end, plan, data = [read(folder/(k+'.json')) for k in ('completion','protocol','results')]
        check(end['results']==sha(folder/'results.json') and end['protocol']==sha(folder/'protocol.json')
              and plan['runner']==sha(ROOT/'scripts'/(script+'.py')) and plan['plan']==sha(ROOT/'docs/batches'/(doc+'.md')),
              stage+' source or results')
        for arm in arms:
            one=folder/arm
            record=read(one/'fit_completed.json')
            check(sha(one/'fit_started.json')==record.get('fit_record',record.get('start'))
                  and record.get('new_supervised_fits',record.get('fits'))==1, stage+' physical fit')
        check(end['model_api_calls']==0 and data['model_api_calls']==0, stage+' usage')
        result[stage]=data
    screen,confirm=result['screen'],result['confirm']
    nets=[v['weighted']-v['original'] for v in screen['folds'].values()]
    check(screen['eligible_for_confirmation']==(min(nets)>=0 and max(nets)>=5), 'screen gate')
    panels=confirm['panels']
    for name in ('A','B','combined'):
        v=panels[name]
        check(v['weighted']-v['regular']==v['gained']-v['lost']==v['delta']['net_correct'], name)
    for field in ('rows','regular','weighted','gained','lost'):
        check(panels['A'][field]+panels['B'][field]==panels['combined'][field], field)
    nets=[panels[k]['delta']['net_correct'] for k in ('A','B')]
    check(confirm['eligible_for_agent']==(min(nets)>=0 and sum(nets)>=8), 'confirmation gate')
    return dict(screen_gain=[screen['folds'][k]['delta']['net_correct'] for k in ('18643','40531')],
                confirmation_gain=nets, upgraded=confirm['eligible_for_agent'], actual_fits=4, api_calls=0)


if __name__=='__main__':
    print(json.dumps(verify()))
