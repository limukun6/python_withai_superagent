"""Audit a deeper encoder's request, local installation, observed token lengths and realized decisions."""
import argparse
from pathlib import Path

from tokenizers import Tokenizer

from superagent.data import load_training_task
from superagent.method_choice import TEXT_DEEP
from superagent.minilm_features import DEEP_FILES, DEEP_MODEL, FILES
from superagent.pretrained_choice import digest
from tools.frontier_study import expect
from tools.pretrained_study import read, train_only, write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('stage','archive','small-weights','deep-weights','request','receipt','output'):
        parser.add_argument('--'+flag,type=Path,required=True)
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
    write(args.output,report)
    print(report['deep_trials'],report['deep_final_wins'],report['correct'],report['texts_over_128'])


if __name__=='__main__':main()
