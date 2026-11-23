"""Verify the public transfer, adaptation and surface results; rebuild grouped counts from local exports."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / 'research/b_quote_transfer_1123'


def read(file):
    return json.loads(file.read_bytes())


def digest(file):
    return hashlib.sha256(file.read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pred(folder, expected, n):
    for k in (0,1):
        file = folder.parent / (folder.name + str(k)) / 'rows.csv'
        require(digest(file) == expected, 'Prediction changed: '+str(file))
    with file.open(encoding='utf-8',newline='') as f:
        rows = list(csv.DictReader(f))
    require([int(r['row_id']) for r in rows]==list(range(n)), 'Rows differ')
    return [int(r['prediction']) for r in rows]


def grouped(groups, truth, first, second):
    require(len(groups)==len(truth)==len(first)==len(second), 'Different prediction shapes')
    out = {}
    for group,y,a,b in zip(groups,truth,first,second):
        counts = out.setdefault(group,[0,0,0,0])
        counts[2*int(a==y)+int(b==y)] += 1
    return [out[k] for k in sorted(out)]


def build(transfer, adaptation):
    originals = read(transfer/'exports.json')
    truth = read(transfer/'scoring_only.json')
    y = truth['winner_left']
    original = pred(transfer/'soft_vote_export_',originals['control'],len(y))
    joint = pred(transfer/'autonomous_export_',originals['joint'],len(y))
    hold = read(adaptation/'scoring_only.json')
    ids = hold['ids']
    adapted = pred(adaptation/'autonomous_export_',read(adaptation/'outer_export.json')['predictions'],len(ids))
    require(hold['labels']==[y[i] for i in ids], 'Frozen and adapted targets differ')
    return dict(columns=['neither','second_only','first_only','both'],
                all_films_control_then_joint=grouped(truth['films'],y,original,joint),
                held_groups_joint_then_adapted=grouped(hold['components'],hold['labels'],[joint[i] for i in ids],adapted))


def verify(pack):
    frozen, adapted, surface = [pack/n for n in ('frozen','adapted','surface')]
    for f in (frozen, adapted, surface):
        end = read(f/'completion.json')
        require(end['protocol']==digest(f/'protocol.json') and end['results']==digest(f/'results.json') and
                end['model_api_calls']==0, 'Invalid completion: '+f.name)
    gathered = read(pack/'group_counts.json')
    require(gathered['columns']==['neither','second_only','first_only','both'],'Cell ordering differs')
    fr, ar = read(frozen/'results.json'),read(adapted/'results.json')
    for key, scores, names, expected in (
            ('all_films_control_then_joint',fr['scores'],('control','joint'),(fr['movies'],fr['joint_only'],fr['control_only'])),
            ('held_groups_joint_then_adapted',ar['scores'],('joint','adapted'),(ar['components'],ar['adapted_only'],ar['joint_only']))):
        table = gathered[key]
        require(all(len(row)==4 and all(type(v) is int and v>=0 for v in row) and sum(row)>0 for row in table),'Invalid group cells')
        cols = list(map(sum,zip(*table)))
        require((len(table),cols[1],cols[2])==expected, 'Grouped disagreements differ')
        require([cols[2]+cols[3],cols[1]+cols[3]]==[scores[n]['correct'] for n in names], 'Grouped accuracy differs')
        require(all(v['correct']/sum(cols)==v['accuracy'] for v in scores.values()),'Accuracy denominator differs')
    fp, ap = read(frozen/'protocol.json'), read(adapted/'protocol.json')
    require(fp['rows']+sum(fp['omitted'].values())==fp['raw_pairs']==2197 and
            ap['train_rows']+ap['held_rows']==fp['rows'] and ap['train_films']+ap['held_films']==fp['movies']
            and fp['headline_overlap_pairs']==ap['shared_outer_texts']==ap['shared_outer_films']==0,'Partition facts differ')
    decision = read(adapted/'model_frozen.json')
    physical=(adapted/'physical.jsonl').read_text().splitlines()
    require(digest(adapted/'physical.jsonl')==decision['fit_log'] and len(physical)==decision['fits']==ar['new_fits']==8
            and decision['selected']==ar['selected'] and decision['actions'][-1]['action']=='stop','Fit or decision records differ')
    require(read(surface/'completion.json')['exports']==digest(surface/'exports.json'),'Surface output seals differ')
    sr = read(surface/'results.json')
    for name, r in sr.items():
        require(r['original_correct']+r['plain_only']-r['original_only']==r['plain_correct']
                and sum(v['rows'] for v in r['word_count'].values())==r['rows']
                and sum(v['correct'] for v in r['word_count'].values())==r['original_correct'], name)
    return dict(frozen=fr['scores']['joint']['correct'], adapted=ar['scores']['adapted']['correct'],
                held_rows=ar['rows'], actual_fits=len(physical), model_api_calls=0,
                swapped_complements={k:v['swapped_complements'] for k,v in sr.items()})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rebuild', nargs=2, type=Path, metavar=('TRANSFER_RUN','ADAPTATION_RUN'))
    parser.add_argument('--pack', type=Path, default=PACK)
    a=parser.parse_args()
    if a.rebuild:
        data=build(*a.rebuild)
        if not (a.pack/'group_counts.json').exists():
            with (a.pack/'group_counts.json').open('x') as f:
                json.dump(data,f,separators=(',',':'))
        require(read(a.pack/'group_counts.json')==data,'Published groups differ from frozen prediction files')
    print(json.dumps(verify(a.pack)))
