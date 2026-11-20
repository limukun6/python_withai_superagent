"""Audit a deeper encoder's request, local installation, observed token lengths and realized decisions."""
import argparse
from collections import Counter
import csv
import io
import json
from pathlib import Path
import shutil
import zipfile

from tokenizers import Tokenizer

from superagent.data import load_training_task
from superagent.method_choice import TEXT_DEEP
from superagent.minilm_features import DEEP_FILES, DEEP_MODEL, FILES
from superagent.pretrained_choice import digest
from tools.frontier_study import expect
from tools.pretrained_study import read, train_only, write
from tools.submission_smoke import command


def portable_predictions(source, weights, destination, deep_weights=None):
    destination.mkdir(parents=True, exist_ok=False)
    moved=destination/'relocated'
    moved.mkdir()
    for name in ('selected_model.joblib','task_spec.json','final_selection.json'):
        shutil.copyfile(source/name,moved/name)
    subjects=('market','railway','museum','river','festival','school','garden','station')
    rows=[]
    for subject in subjects:
        pair=(f'An update on the town {subject}',f'What changed at the {subject} this week?')
        rows.extend([pair,pair[::-1]])
    content=io.StringIO()
    writer=csv.writer(content)
    writer.writerow(['headline_A','headline_B'])
    writer.writerows(rows)
    package=destination/'synthetic.zip'
    with zipfile.ZipFile(package,'w') as archive:archive.writestr('test.csv',content.getvalue())
    usage={}
    checksums=[]
    per_model={}
    flags=['--minilm-weights',weights]
    if deep_weights is not None:
        flags.extend(['--deep-minilm-weights',deep_weights])
    def model_tokens(output):
        ledger=destination/(output.name+'.features')/'downstream_usage.jsonl'
        counts=Counter()
        if ledger.exists():
            for line in ledger.read_text().splitlines():
                event=json.loads(line)
                counts[event['model']]+=event['input_tokens']
        return counts
    for label,model,filename in (('original',source,'original.csv'),('relocated',moved,'relocated.csv'),('cached',moved,'relocated.csv')):
        output=destination/filename
        before=model_tokens(output)
        command(destination/(label+'.log'),'predict',package,'--run-dir',model,*flags,'--output',output)
        manifest=read(destination/'prediction_manifest.json')
        expect(manifest['prediction_count']==len(rows) and not manifest['test_labels_read'],'Synthetic prediction manifest changed')
        usage[label]=manifest['prediction_feature_usage']
        per_model[label]=dict(model_tokens(output)-before)
        expect(sum(per_model[label].values())==usage[label]['input_tokens'],'Model-level token counts differ')
        checksums.append(digest(output))
    expect(len(set(checksums))==1 and usage['original']['input_tokens']==usage['relocated']['input_tokens']>0
           and usage['cached']['input_tokens']==0,'Moved or cached prediction differs')
    if deep_weights is not None:
        expect(len(per_model['original'])==2 and per_model['original']==per_model['relocated'],
               'Both components must run on new examples after relocation')
    with output.open() as f:predictions=list(csv.DictReader(f))
    expect([int(row['row_id']) for row in predictions]==list(range(len(rows))),'Synthetic rows changed order')
    flipped=sum(a['prediction']!=b['prediction'] for a,b in zip(predictions[::2],predictions[1::2],strict=True))
    return dict(rows=len(rows),swapped_pairs=len(subjects),opposite_predictions=flipped,prediction_sha256=checksums[0],
                model_sha256=digest(moved/'selected_model.joblib'),usage=usage,input_tokens_by_model=per_model)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('stage','archive','small-weights','deep-weights','request','receipt','output'):
        parser.add_argument('--'+flag,type=Path,required=True)
    parser.add_argument('--portability-output',type=Path,help='Use an exported L12 classifier on a new synthetic unlabeled prediction package')
    args=parser.parse_args()
    protocol=read(args.stage/'protocol.json')
    results=read(args.stage/'results.json')
    request=read(args.request)
    rec=read(args.receipt)
    expect(rec['request_sha256']==digest(args.request) and rec['model']==DEEP_MODEL and rec['api_rmb']==0,'Wrong weight receipt')
    for directory,files in ((args.small_weights,FILES),(args.deep_weights,DEEP_FILES)):
        for n,(_,sha,size) in files.items():
            expect((directory/n).stat().st_size==size and digest(directory/n)==sha,'Encoder checkpoint differs')
    expect(digest(args.archive)==protocol['archive_sha256'],'Archive differs')
    opened=[]
    with train_only(args.archive,opened):task=load_training_task(args.archive)
    ids=sorted({i for fold in protocol['indices'] for i in fold['outer']})
    texts=sorted({s for i in ids for s in task.X_train[i]})
    tokenizer=Tokenizer.from_file(str(args.small_weights/'tokenizer.json'))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    lengths=[len(t.ids) for t in tokenizer.encode_batch(texts,add_special_tokens=True)]
    folds=[]
    for i,paths in enumerate(protocol['locations']):
        source=Path(paths['encoder-depth'])
        row=results['folds'][i]['arms']['encoder-depth']
        log=read(source/'method_decision_7.json')
        ev=log['evidence']
        direct=log.get('candidate')==TEXT_DEEP
        expect(direct==(row['extras'][1]==TEXT_DEEP),'Recorded and executed depth choices differ')
        trials=read(source/'trials.json')
        local=[dict(candidate=t['candidate'],search_accuracy=t['search_accuracy']) for t in trials[5:]]
        folds.append(dict(fold=protocol['fold_numbers'][i],chose_deeper=direct,selected_deeper=row['winner']==TEXT_DEEP,
                          corrected=ev['corrected'],lost=ev['lost'],remaining_shared_errors=ev['residual_shared_errors'],
                          seconds_estimate=2*trials[5]['worker_wall_seconds'],search=local,winner=row['winner']))
    report=dict(archive_sha256=protocol['archive_sha256'],stage_sha256=digest(args.stage/'results.json'),
                requested_by_training_replay=sum(d['selected']==TEXT_DEEP for d in request['decisions']),
                deep_trials=sum(f['chose_deeper'] for f in folds),deep_final_wins=sum(f['selected_deeper'] for f in folds),
                public_weight_download=rec,local_on_disk_bytes=sum(item[2] for item in DEEP_FILES.values()),
                pool_rows=len(ids),unique_texts=len(texts),maximum_tokens=max(lengths),
                texts_over_128=sum(length>128 for length in lengths),texts_over_256=sum(length>256 for length in lengths),
                common_tokenizer_sha256=FILES['tokenizer.json'][1],folds=folds,correct=results['correct'],
                training_members=opened)
    if args.portability_output:
        selected=next(i for i,fold in enumerate(folds) if fold['selected_deeper'])
        report['portable']=portable_predictions(Path(protocol['locations'][selected]['encoder-depth']),
                                                args.deep_weights.resolve(),args.portability_output.resolve())
    write(args.output,report)
    print(report['deep_trials'],report['deep_final_wins'],report['correct'],report['texts_over_128'])


if __name__=='__main__':main()
