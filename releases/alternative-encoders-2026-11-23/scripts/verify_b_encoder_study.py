"""Verify archived encoder-study decisions, row counts, resource totals and frozen exports."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'research/b_alternative_encoders_1123'


def get(name):
    return json.loads((DATA/name).read_bytes())


def require(condition, label):
    if not condition:
        raise ValueError(label)


def main():
    mainline=get('native/results.json')
    first=get('full_vote/results.json')
    last=get('materialized_vote/results.json')
    probe=get('probe.json')
    require(probe['order']==['deep-e5','deep-bge'] and probe['titles']==512,'Unlabeled preflight')
    main_fit=0
    native=get('native/exports_frozen.json')['exports']
    for arm,r in mainline['results'].items():
        receipt=get('native/'+arm+'_receipt.json')
        require(r['actual_fits']==receipt['actual_fits'] and receipt['selected']==r['selected'] and not r['selected_new'],
                'Native delivery identity')
        require(r['correct']==1224 and native[arm]==native['deep'],'Native prediction equality')
        main_fit+=receipt['actual_fits']
    follow_fits=sum(v['new_fits'] for v in get('full_vote/fits.json').values())
    last_fit=get('materialized_vote/worker_fits.json')['completed_classifier_fits']
    require(main_fit==20 and follow_fits==2 and last_fit==1 and main_fit+follow_fits+last_fit==23,'Physical accounting')
    for name,results in [('full_vote',first),('materialized_vote',last)]:
        require(hashlib.sha256((DATA/name/'exports_frozen.json').read_bytes()).hexdigest()==results['export_seal'],'Export seal')
        for row in results['reserve'].values():
            require(row['rows']==1693 and row['correct']==row['reference_correct']+row['gained']-row['lost'],'Paired correctness')
    replay=get('scheduler_replay/results.json')
    require(replay['decision_seal']==hashlib.sha256((DATA/'scheduler_replay/decisions_before_score_lookup.json').read_bytes()).hexdigest(),
            'Decision seal')
    require(all(r['fit_equivalents']==8 and r['new_physical_fits']==0 for r in replay['results'].values()),'Replay accounting')
    require(first['reserve']['soft_vote']==last['reserve']['e5_vote'] and last['path_reused_fits']==7,
            'Frozen E5 comparator or BGE reuse')
    require(mainline['model_api']==first['model_api_requests']==last['model_api']==0,'API accounting')
    return dict(native_correct=mainline['control_correct'],e5_vote_correct=first['reserve']['soft_vote']['correct'],
        bge_vote_correct=last['reserve']['bge_vote']['correct'],rows=1693,new_fits=23,model_api=0)


if __name__=='__main__':
    print(json.dumps(main(),indent=2))
