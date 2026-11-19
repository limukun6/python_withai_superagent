"""Produce a shareable Task1 comparison from verified runs, without course rows or model files."""
import argparse
from pathlib import Path

from superagent.method_choice import select_action
from superagent.pretrained_choice import digest
from tools.pretrained_study import read, write


def compact(root):
    protocol=read(root/'protocol.json')
    results=read(root/'results.json')
    verification=root/('verification.json' if 'paired' in results['correct'] else 'decision_and_device_audit.json')
    audit=read(verification)
    target=protocol['order'][0][-1]
    rows_seen=set()
    folds=[]
    decisions_replayed=0
    for index,fold in enumerate(results['folds']):
        predictions=root/f'fold_{index}'/'outer_predictions.json'
        held=read(predictions)['rows']
        assert held==protocol['indices'][index]['outer'] and not rows_seen.intersection(held)
        rows_seen.update(held)
        arms={}
        for name,run in fold['arms'].items():
            for record in run['decisions']:
                chosen,_=select_action(record['evidence'],record['eligible'],name,record['previous_methods'])
                assert chosen==(record['candidate'] if record['decision']=='try' else None)
                decisions_replayed+=1
            arms[name]=dict(correct=run['correct'],final_model=run['winner'],fits=run['fits'],
                extra_trials=[t['id'] for t in run['trials'][5:]],
                devices=[t['train_metadata']['device_decision']['selected'] for t in run['trials'] if
                         t['train_metadata'] and 'device_decision' in t['train_metadata']],
                final_model_sha256=run['sha256'],minilm=run['downstream_minilm'],qwen=run['usage'])
        folds.append(dict(fold=protocol.get('fold_numbers',list(range(1,len(results['folds'])+1)))[index],
                          rows=len(held),arms=arms,changes=fold[target+'_vs'],
                          outer_predictions_sha256=digest(predictions)))
    assert len(rows_seen)==results['rows'] and audit['counts']['fits']==sum(v['fits'] for f in folds for v in f['arms'].values())
    return dict(rows=results['rows'],correct=results['correct'],accuracy=results['accuracy'],wins=results['wins'],
                passed=results['promote'],folds=folds,verification_sha256=digest(verification),
                protocol_sha256=digest(root/'protocol.json'),results_sha256=digest(root/'results.json'),
                usage_and_replay_counts=audit['counts'],current_release_replayed_decisions=decisions_replayed,
                remote_generation_calls=audit['api_generation_requests'],actual_api_charge_rmb=audit['actual_api_charge_rmb'],
                qwen_reference_rmb=results['course_rmb']), rows_seen, protocol


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('initial','paired','remaining','smoke','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    first,pool,p1=compact(args.initial)
    development,seen,p2=compact(args.paired)
    replication,held,p3=compact(args.remaining)
    assert not seen.intersection(held) and seen|held==pool
    assert p2['prior_protocol_sha256']==first['protocol_sha256'] and p3['prior_protocol_sha256']==development['protocol_sha256']
    assert p1['archive_sha256']==p2['archive_sha256']==p3['archive_sha256']
    assert all(v==p3['source'][k] for k,v in p2['source'].items() if k.startswith('superagent/'))
    folds=development['folds']+replication['folds']
    combined={arm:sum(f['arms'][arm]['correct'] for f in folds) for arm in development['correct']}
    paired={}
    for arm in ('classic','adaptive'):
        delta=[f['arms']['paired']['correct']-f['arms'][arm]['correct'] for f in folds]
        paired[arm]=dict(wins=sum(v>0 for v in delta),draws=delta.count(0),losses=sum(v<0 for v in delta),
                         corrected=sum(f['changes'][arm]['corrected'] for f in folds),
                         lost=sum(f['changes'][arm]['lost'] for f in folds))
    device=read(args.initial/'decision_and_device_audit.json')['device_check']
    stages=(first,development,replication)
    cost=sum(stage['qwen_reference_rmb'] for stage in stages)
    assert cost<=p1['campaign_cap']==p2['campaign_cap']==p3['campaign_cap']
    smoke={key:{k:v for k,v in row.items() if k!='decisions'} for key,row in read(args.smoke/'smoke_summary.json').items()}
    report=dict(initial=first,first_three=development,last_two=replication,
                all_paired_folds=dict(rows=len(pool),correct=combined,accuracy={k:v/len(pool) for k,v in combined.items()},
                                      paired_vs=paired),api_rmb=sum(s['actual_api_charge_rmb'] for s in stages),
                remote_generation_calls=sum(s['remote_generation_calls'] for s in stages),
                qwen_local_reference_rmb=cost,campaign_cap_rmb=p1['campaign_cap'],
                device_diagnostic=device,anonymous_smoke=smoke)
    write(args.output,report)
    print(report['all_paired_folds'],flush=True)


if __name__=='__main__':main()
