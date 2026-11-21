"""Measure whether group-aware paired accuracy feedback changes an action on actual histories."""
import argparse
import gzip
import json
import math
from pathlib import Path
from unittest.mock import patch
import zipfile

import numpy as np

from scripts import replay_b1_observed_actions as replay, run_b1_ab_acceptance as rec
from scripts.prepare_b_ordered_relation import families, SOURCE_SHA
from superagent import ab_run, ab_scheduler
from superagent.ab_comparison import MIN_GAIN
from superagent.data import load_development_task
from superagent.metrics import score_of
from superagent.scheduler_bridge import fingerprint


def relation_groups(source, case):
    ids_data = gzip.decompress((rec.ROOT/'research/br_01_relation_groups/source_ids.json.gz').read_bytes())
    expected = rec.read(rec.ROOT/'research/br_01_relation_groups/results.json')
    if ab_run.sha(source)!=SOURCE_SHA or rec.sha(ids_data)!=expected['outer_source_ids_sha256']:
        raise ValueError('Source or registered optimization IDs changed')
    ordered = json.loads(ids_data)['train']
    wanted, records = set(ordered), {}
    with zipfile.ZipFile(source) as archive, archive.open('SICK.txt') as lines:
        header = lines.readline().decode('utf-8-sig').rstrip('\r\n').split('\t')
        cols = [header.index(k) for k in ('sentence_A','sentence_B','sentence_A_original','sentence_B_original')]
        key = header.index('pair_ID')
        if header[-1]!='SemEval_set':
            raise ValueError('Expected final publisher split field')
        for raw in lines:
            raw=raw.rstrip(b'\r\n')
            if raw.rsplit(b'\t',1)[-1]!=b'TRAIN':
                continue
            fields=raw.split(b'\t')
            number=int(fields[key])
            if number in wanted:
                a,b,origin_a,origin_b=(fields[k].decode() for k in cols)
                records[number]=(number,a,b,None,origin_a,origin_b)
    rows=[records[n] for n in ordered]
    data=load_development_task(case/'train.zip')
    group=families(rows)
    if ([list(r[1:3]) for r in rows]!=data.X_train.values.tolist() or
            len(set(group))!=expected['outer']['train_families']):
        raise ValueError('Groups, source rows or text order differ from BR-01')
    return group


def diagnostics(lead, other, trusted, first, second):
    a,b,y=(np.asarray(v) for v in (first,second,trusted['labels']))
    gap=score_of(lead)-score_of(other)
    wins=(a==y).astype(int)-(b==y).astype(int)
    if not math.isclose(float(wins.mean()),gap,abs_tol=1e-10):
        raise ValueError('Scores and paired correctness disagree')
    _, groups=np.unique(trusted['groups'],return_inverse=True)
    sizes=np.bincount(groups)
    if len(sizes)<2:
        raise ValueError('At least two independent groups are required for the diagnostic')
    centered=np.bincount(groups,weights=wins)-sizes*wins.mean()
    spread=math.sqrt(len(sizes)/(len(sizes)-1)*float(centered@centered))/len(y)
    disagreement=int(np.count_nonzero(a!=b))
    old=max(MIN_GAIN,math.sqrt(disagreement)/len(y))
    clustered=max(MIN_GAIN,spread)
    return dict(lead=lead['candidate'],other=other['candidate'],rows=len(y),groups=len(sizes),
                discordant=disagreement,both_wrong_with_different_outputs=int(np.count_nonzero((a!=b)&(a!=y)&(b!=y))),
                score_gap=gap,old_threshold=old,grouped_threshold=clustered,
                old_eligible=bool(disagreement and gap<=old),grouped_eligible=bool(disagreement and gap<=clustered))


def compare(case, entry, frozen, source=None):
    caps,trusted=replay.trusted_input(case,entry,frozen)
    official=case/'run'
    manifest=rec.read(official/'run_manifest.json')
    parts=('ab_scheduler.py','ab_comparison.py')
    if any(manifest['source_files'][k]!=ab_run.sources()[k] for k in parts):
        raise ValueError('Scheduler or promotion code differs from the recorded path')
    actual=rec.read(official/'actions.json')
    logs=[json.loads(line) for line in (official/'events.jsonl').read_text().splitlines()]
    events=[e for e in logs if e['kind']=='scheduler_action_admitted']
    with_groups={k:dict(v) for k,v in trusted.items()}
    if source is not None:
        groups=relation_groups(source,case)
        for s,v in with_groups.items():
            v['groups']=groups[v['row_ids']].tolist()
    pairs,changed={},[]
    first_difference=None
    def clustered(lead,front,records,metric,details):
        if metric.name!='accuracy' or details is None:
            raise ValueError('This preregistration requires paired, labelled optimization accuracy')
        for other in front:
            if lead['trial']==other['trial']:
                continue
            row=diagnostics(lead,other,details,records[lead['trial']],records[other['trial']])
            pairs[lead['candidate'],other['candidate']]=row
            if row['grouped_eligible']:
                yield lead,other,row['discordant'],row['rows'],'grouped paired correctness'
    for i,(event,old) in enumerate(zip(events,actual,strict=True),1):
        state=rec.read(official/f'state_{i}.json')
        if fingerprint(state)!=event['state_sha256'] or old!=event['action']:
            raise ValueError('The admitted state or action differs from execution')
        original=ab_scheduler.next_action(state,trusted_partitions=trusted,capabilities=caps)
        if original!=old:
            raise ValueError(f'Unchanged policy failed replay at decision {i}')
        with patch.object(ab_scheduler,'_contenders',clustered):
            new=ab_scheduler.next_action(state,trusted_partitions=with_groups,capabilities=caps)
        if replay.key(new)!=replay.key(old):
            diff=dict(step=i,original=replay.key(old),grouped=replay.key(new))
            changed.append(diff)
            if first_difference is None:
                first_difference=diff
    return dict(decisions=len(actual),original_actions_exact=len(actual),
        unique_examined_directed_pairs=len(pairs),threshold_crossings=sum(v['old_eligible']!=v['grouped_eligible'] for v in pairs.values()),
        primary_rows=len(trusted['primary']['row_ids']),primary_groups=len(set(with_groups['primary']['groups'])),
        first_difference=first_difference,changed_on_original_states=changed,pairs=list(pairs.values()))


def run(args):
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    rec.offline(output)
    plan=rec.ROOT/'docs/batches/BA19_04_feedback_group_plan.md'
    qp,rp=rec.read(args.release/'protocol.json'),rec.read(args.relation/'protocol.json')
    packets=rec.read(args.release/'preflight.json')
    rec.write(output/'protocol.json',dict(batch='BA19-04',plan_sha256=ab_run.sha(plan),
        driver_sha256=ab_run.sha(Path(__file__)),Released_protocol_sha256=ab_run.sha(args.release/'protocol.json'),
        relation_protocol_sha256=ab_run.sha(args.relation/'protocol.json'),fits=0,api_cap=0,outer_scores=0))
    try:
        results={}
        for name,scope in qp['order']:
            p=packets[name]
            entry=dict(files={'train.zip':p['train_sha256']},catalog_sha256=p['catalogs'][scope]['sha256'],splits=p['splits'])
            results[name+'_'+scope]=compare(args.release/(name+'_'+scope),entry,qp['source'])
        ref=rec.read(rec.ROOT/'research/br_01_relation_groups/results.json')
        more=rec.read(rec.ROOT/'research/br_03_ordered_features/preflight.json')
        for scope in ('baseline','full'):
            digest=rec.digest(ref['menu']) if scope=='baseline' else more['catalog_sha256']
            entry=dict(files={'train.zip':ref['train_sha256']},catalog_sha256=digest,splits=more['splits'])
            results['relation_'+scope]=compare(args.relation/scope,entry,rp['source_sha256'],args.relation_source)
        rec.write(output/'results.json',results)
        rec.write(output/'completion.json',dict(status='completed',paths=len(results),reproduced_decisions=sum(r['decisions'] for r in results.values()),
            paths_with_first_difference=sum(r['first_difference'] is not None for r in results.values()),
            fits=0,model_predictions=0,model_api_requests=0,outer_scores_read=0))
    except Exception as error:
        rec.write(output/'completion.json',dict(status='failed',error=str(error),fits=0,model_api_requests=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('relation','release','relation-source','output'):
        parser.add_argument('--'+flag,type=Path,required=True)
    run(parser.parse_args())
