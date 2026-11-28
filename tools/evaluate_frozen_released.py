"""Project official features to anonymous packets, seal the four frozen predictions, then score."""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
import pyarrow as pa
from pyarrow import csv as arrow_csv

from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT, archive, make_zip, read, require
from superagent.first_delivery import digest


RECEIPT = 'research/unified_environment_freeze_1128/frozen_release.json'
CASES = [('Task1','table'),('Task2','titles')]
ARMS = ('train_only','train_plus_validation')


def once(path, value):
    if path.exists():
        require(read(path)==value,'Recorded output differs: '+path.name)
        return
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,delete=False) as stream:
        json.dump(value,stream,indent=2,ensure_ascii=False,allow_nan=False)
        stream.write('\n')
    os.replace(stream.name,path)


def committed_freeze(args):
    require(len(args.frozen_commit)==40,'Supply the full pre-test freeze commit')
    def git(*command):
        return subprocess.check_output(['git','-C',str(ROOT),*command])
    require(git('rev-parse','HEAD').decode().strip()==args.frozen_commit,'Use the original freeze checkout')
    require(git('show',args.frozen_commit+':'+RECEIPT)==(ROOT/RECEIPT).read_bytes(),'Freeze record differs')
    script=Path(__file__).resolve().relative_to(ROOT).as_posix()
    require(git('show',args.frozen_commit+':'+script)==Path(__file__).read_bytes(),'Scoring program differs')
    release=read(ROOT/RECEIPT)
    require(release['solver']==source_identity(ROOT) and release['primary_arm']=='train_only'
            and release['scoring_program_sha256']==digest(Path(__file__).resolve())
            and release['formal_test_protocol_sha256']==digest((ROOT/RECEIPT).with_name('formal_test_protocol.json'))
            and release['designated_validation_complete'] and not release['official_test_opened_at_freeze']
            and set(release['arms'])==set(ARMS),'Model or freeze identity differs')
    for arm, path in zip(ARMS,(args.bundle,args.augmented_bundle)):
        ref=release['arms'][arm]
        require(ref['archive_sha256']==digest(path) and set(ref['verified'])=={n for n,_ in CASES},
                'The '+arm+' model bundle differs from its freeze')
    for name,_ in CASES:
        require(len({release['arms'][a]['verified'][name]['selected_model'] for a in ARMS})==2,
                'Both fits must have distinct model identities')
    return release


def member(z, name, *, optional=False):
    names=[n for n in z.namelist() if '__MACOSX' not in Path(n).parts and Path(n).name==name]
    if optional and not names:
        return None
    require(len(names)==1,'Expected one official course member named '+name)
    return names[0]


def projected_csv(z, fields):
    names=list(fields)
    with z.open(member(z,'test.csv')) as stream:
        return arrow_csv.read_csv(stream,read_options=arrow_csv.ReadOptions(use_threads=False),
                    parse_options=arrow_csv.ParseOptions(newlines_in_values=True),
                    convert_options=arrow_csv.ConvertOptions(include_columns=names,column_types=fields,
                                                             strings_can_be_null=False))


def feature_packet(source, tag, target):
    with zipfile.ZipFile(source) as z:
        if tag=='table':
            data=z.read(member(z,'C_test.npy'))
            stream=io.BytesIO(data)
            version=np.lib.format.read_magic(stream)
            readers={(1,0):np.lib.format.read_array_header_1_0,(2,0):np.lib.format.read_array_header_2_0}
            require(version in readers,'Unexpected NPY header')
            shape,_,_=readers[version](stream)
            require(len(shape)==2 and shape[0]>0,'Expected a nonempty table')
            entries=[('C_test.npy',data)]
            count=shape[0]
        else:
            # The distributed CSV also holds labels. Arrow materialises just these two feature columns.
            table=projected_csv(z,{'headline_A':pa.string(),'headline_B':pa.string()})
            count=table.num_rows
            buf=io.StringIO(newline='')
            writer=csv.writer(buf)
            writer.writerow(table.column_names)
            writer.writerows(zip(*(c.to_pylist() for c in table.columns)))
            entries=[('test.csv',buf.getvalue().encode('utf-8'))]
    make_zip(target,entries)
    return count


def prediction_rows(path, count):
    with path.open(encoding='utf-8-sig',newline='') as source:
        reader=csv.DictReader(source)
        require(reader.fieldnames==['row_id','prediction'],'Unexpected prediction columns')
        predictions=[]
        for i,row in enumerate(reader):
            require(row['row_id']==str(i) and row['prediction'] in ('0','1'),'Invalid row identity or class')
            predictions.append(int(row['prediction']))
    require(len(predictions)==count and count>0,'Incomplete predictions')
    return np.asarray(predictions,dtype=np.int64)


def bundle_digests(release):
    return {arm:release['arms'][arm]['archive_sha256'] for arm in ARMS}


def predict(args, release):
    args.out.mkdir(parents=True,exist_ok=True)
    stamp=args.out/'freeze_before_test.json'
    archives=bundle_digests(release)
    if not stamp.exists():
        once(stamp,dict(frozen_commit=args.frozen_commit,archives=archives,solver=release['solver'],
                        primary_arm=release['primary_arm'],recorded_before_first_feature_read=datetime.now(timezone.utc).isoformat()))
    registered=read(stamp)
    require(registered['frozen_commit']==args.frozen_commit and registered['archives']==archives,'Another release owns this output')
    seal=args.out/'predictions_sealed.json'
    if seal.exists():
        check_sealed(args,release)
        return
    env={k:os.environ[k] for k in ('PATH','TMPDIR','TMP','TEMP','SYSTEMROOT','WINDIR') if k in os.environ}
    env.update(OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    cases={}
    with tempfile.TemporaryDirectory(dir=args.out) as temp:
        work=Path(temp)
        for arm,path in zip(ARMS,(args.bundle,args.augmented_bundle)):
            with zipfile.ZipFile(path) as z:
                z.extractall(work/arm)
        for name,tag in CASES:
            cases[name]={}
            packet=work/(name+'_features.zip')
            count=feature_packet(archive(args.packages,tag),tag,packet)
            feature_sha=digest(packet)
            for arm in ARMS:
                output=args.out/arm/name
                output.mkdir(parents=True,exist_ok=True)
                receipt=output/'prediction.json'
                prediction=output/'predictions.csv'
                selected=release['arms'][arm]['verified'][name]['selected_model']
                if not receipt.exists():
                    solver=work/arm/name
                    require(read(solver/'solution/bundle_manifest.json')['model_sha256']==selected,
                            'Unexpected frozen model for '+name+' '+arm)
                    temporary=work/(arm+'_'+name)
                    temporary.mkdir()
                    result=temporary/'predictions.csv'
                    started=time.monotonic()
                    with (output/'predict.log').open('w',encoding='utf-8') as log:
                        subprocess.run([sys.executable,'-I',str(solver/'predict.py'),'--features',str(packet),
                                        '--output',str(result)],cwd=temporary,env=env,stdin=subprocess.DEVNULL,
                                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
                    prediction_rows(result,count)
                    os.replace(result,prediction)
                    once(receipt,dict(rows=count,feature_packet_sha256=feature_sha,prediction_sha256=digest(prediction),
                                      model_sha256=selected,inference_seconds=round(time.monotonic()-started,3),
                                      model_api=0,classifier_fits=0))
                rec=read(receipt)
                require(rec['model_sha256']==selected and rec['prediction_sha256']==digest(prediction)
                        and rec['rows']==count and rec['feature_packet_sha256']==feature_sha,'Existing inference differs')
                cases[name][arm]=rec
    once(seal,dict(frozen_commit=args.frozen_commit,archives=archives,cases=cases,
                   sealed_before_label_scoring=datetime.now(timezone.utc).isoformat()))


def check_sealed(args,release):
    seal=read(args.out/'predictions_sealed.json')
    require(seal['frozen_commit']==args.frozen_commit and seal['archives']==bundle_digests(release)
            and set(seal['cases'])=={n for n,_ in CASES},'Both tasks must be sealed')
    for name,_ in CASES:
        pair=seal['cases'][name]
        require(set(pair)==set(ARMS) and len({v['rows'] for v in pair.values()})==1
                and len({v['feature_packet_sha256'] for v in pair.values()})==1,'Both arms must use identical rows and features')
        for arm,case in pair.items():
            path=args.out/arm/name/'predictions.csv'
            require(case==read(path.with_name('prediction.json')) and
                    case['model_sha256']==release['arms'][arm]['verified'][name]['selected_model'] and
                    case['prediction_sha256']==digest(path),'Frozen prediction differs')
            prediction_rows(path,case['rows'])
    return seal


def labels_from_archive(source,tag):
    with zipfile.ZipFile(source) as z:
        if tag=='table':
            target=member(z,'y_test.npy',optional=True)
            return np.load(io.BytesIO(z.read(target)),allow_pickle=False).reshape(-1) if target else None
        try:
            return np.asarray(projected_csv(z,{'label_pairwise':pa.int64()}).column(0))
        except pa.ArrowKeyError:
            return None


def score(args,release):
    seal=check_sealed(args,release)
    identity=digest(args.out/'predictions_sealed.json')
    score_file=args.out/'official_scores.json'
    if not score_file.exists():
        results={}
        for name,tag in CASES:
            pair=seal['cases'][name]
            count=pair[ARMS[0]]['rows']
            labels=labels_from_archive(archive(args.packages,tag),tag)
            if labels is None:
                results[name]=dict(status='labels_unavailable',metric='accuracy',rows=count)
                continue
            require(labels.shape==(count,) and set(np.unique(labels)).issubset({0,1}),'Label shape or domain differs')
            masks={}
            arms={}
            for arm in ARMS:
                masks[arm]=prediction_rows(args.out/arm/name/'predictions.csv',count)==labels
                correct=int(masks[arm].sum())
                arms[arm]=dict(correct=correct,accuracy=correct/count,prediction_sha256=pair[arm]['prediction_sha256'])
            base,extra=(masks[a] for a in ARMS)
            gained,lost=int((~base & extra).sum()),int((base & ~extra).sum())
            results[name]=dict(metric='accuracy',rows=count,arms=arms,paired=dict(
                both_correct=int((base & extra).sum()),both_wrong=int((~base & ~extra).sum()),
                corrected_by_adding_validation=gained,regressed_after_adding_validation=lost,
                net_correct=gained-lost,delta_percentage_points=100*(gained-lost)/count))
        once(score_file,dict(frozen_commit=args.frozen_commit,primary_arm=release['primary_arm'],
                            seal_sha256=identity,results=results,scored_at_utc=datetime.now(timezone.utc).isoformat(),
                            subsequent_training=False,additional_fits=0,model_api=0))
    result=read(score_file)
    require(result['seal_sha256']==identity and result['frozen_commit']==args.frozen_commit,'Score belongs to another seal')
    delivery=args.out/'Released-Task1-Task2-Official-Predictions.zip'
    expected=[f'{arm}/{name}/predictions.csv' for arm in ARMS for name,_ in CASES]
    expected+=['freeze_before_test.json','predictions_sealed.json','official_scores.json']
    if not delivery.exists():
        with tempfile.TemporaryDirectory(dir=args.out) as tmp:
            stage=Path(tmp)/delivery.name
            make_zip(stage,[(n,(args.out/n).read_bytes()) for n in expected])
            os.replace(stage,delivery)
    with zipfile.ZipFile(delivery) as z:
        require(z.namelist()==expected and all(z.read(n)==(args.out/n).read_bytes() for n in expected),
                'Official delivery differs from sealed files')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['predict','score'])
    parser.add_argument('--frozen-commit',required=True)
    for arg in ('bundle','augmented-bundle','packages','out'):
        parser.add_argument('--'+arg,required=True,type=lambda s:Path(s).resolve())
    arguments=parser.parse_args()
    manifest=committed_freeze(arguments)
    arguments.out.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(arguments.out/'.evaluation.lock.sqlite',timeout=1) as lock:
        lock.execute('BEGIN IMMEDIATE')
        {'predict':predict,'score':score}[arguments.phase](arguments,manifest)
