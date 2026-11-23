"""Check original feedback, physical fits, two external decisions and cross-checkout predictions."""
from hashlib import sha256
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'research/b_single_conflict_1123'


def read(f): return json.loads(f.read_bytes())
def sha(f): return sha256(f.read_bytes()).hexdigest()


def require(v, message):
    if not v:
        raise ValueError(message)


def verify():
    d,r=PACK/'diagnostic',PACK/'replication'
    for p in (d,r):
        end=read(p/'completion.json')
        for k,n in ((('result' if p==d else 'results'),'results'),('protocol','protocol'),('exports','exports'),
                    (('decision' if p==d else 'decisions'),('decision' if p==d else 'decisions'))):
            require(end[k]==sha(p/(n+'.json')),n)
    first,rep=read(d/'results.json'),read(r/'results.json')
    decision=read(d/'decision.json')
    require(sha(d/'physical.jsonl')==decision['actual']['physical'] and
            len((d/'physical.jsonl').read_text().splitlines())==decision['actual']['fits']==first['new_fits']==10,'diagnostic fits')
    check=decision['decision']['single_conflict']
    require(check['status']==first['status']=='switch' and check['fresh_net_correct']>0 and check['unique_gain']>=.002
            and first['new']-first['prior']==first['gained']-first['lost']==9,'diagnostic decision')
    rp=read(r/'protocol.json')
    require(rp['train_rows']+rp['outer_rows']==rp['available']==1744 and
            rp['historical_outer']==450 and rp['outer_shared_films']==rp['outer_shared_texts']==0,'fresh split')
    rr=read(r/'new_review.json')['single_conflict']
    require(rr['status']==rep['conflict']=='retain' and rr['fresh_net_correct']<0 and rep['correct_old']==rep['correct_new']
            and not rep['old_reused'] and read(r/'exports.json')['new']==read(r/'exports.json')['old'],'replication decision')
    arms=read(r/'decisions.json')
    for arm in ('new','old'):
        f=r/(arm+'.physical.jsonl')
        require(sha(f)==arms[arm]['physical'] and len(f.read_text().splitlines())==arms[arm]['fits'],arm+' physical receipt')
    integration=read(PACK/'a117/receipt.json')
    require(integration['predictions']==read(d/'exports.json')['new'] and integration['rows']==first['rows'] and
            integration['actions']==len(decision['actual']['actions']),'teammate integration')
    require(len(read(d/'title_replay.json'))==first['prior_titles_replayed'] and
            first['model_api_calls']==rep['model_api_calls']==integration['model_api_calls']==0,'resource receipts')
    return dict(diagnostic=(first['prior'],first['new']),replication=(rep['correct_old'],rep['correct_new']),
                new_fits=first['new_fits']+rep['actual_new_fits'],model_api_calls=0,a117_verified_rows=integration['rows'])


if __name__=='__main__':
    print(json.dumps(verify()))
