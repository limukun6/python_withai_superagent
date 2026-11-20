"""Separate a train-feedback weight request from a fixed-URL public download."""
import argparse
import os
from pathlib import Path
from unittest.mock import patch

import joblib

from superagent.method_choice import TEXT_DEEP, TEXT_MINILM, plan
from superagent.minilm_features import DEEP_FILES, DEEP_MODEL
from superagent.pretrained_choice import acquire, digest
from tools.frontier_study import expect, receipt
from tools.pretrained_study import read
from tools.soft_vote_diagnostic import freeze_file


def main():
    p=argparse.ArgumentParser(description=__doc__)
    commands=p.add_subparsers(dest='command',required=True)
    decide=commands.add_parser('decide')
    for flag in ('reference','weights','request'):decide.add_argument('--'+flag,type=Path,required=True)
    decide.add_argument('--arm',required=True)
    fetch=commands.add_parser('fetch')
    for flag in ('weights','request','record'):fetch.add_argument('--'+flag,type=Path,required=True)
    args=p.parse_args()
    pins={k:v[1] for k,v in DEEP_FILES.items()}
    if args.command=='decide':
        ref=read(args.reference/'protocol.json')
        evidence=[]
        with patch.dict(os.environ,{'SUPERAGENT_OFFLINE':'0'}), patch('requests.sessions.Session.request',side_effect=AssertionError('Network in decision replay')):
            for i,paths in enumerate(ref['locations']):
                source=Path(paths[args.arm])
                rec=receipt(source)
                runtime=read(source/'pretrained_runtime.json')
                payload=joblib.load(source/'inner_data.joblib')
                history=read(source/'trials.json')
                cfg=dict(minilm_weights=runtime['minilm_weights'],deep_minilm_weights=str(args.weights.resolve()),
                         trial_timeout_seconds=ref['options']['trial_timeout'])
                first=plan(payload['kind'],history[:5],source,payload,cfg,'encoder-depth')
                expect(first.get('candidate')==rec['extras'][0]==TEXT_MINILM,'The first experiment differs')
                action=plan(payload['kind'],history[:6],source,payload,cfg,'encoder-depth')
                ev=action['evidence']
                evidence.append(dict(fold=ref['fold_numbers'][i],selected=action['candidate'],corrected=ev['corrected'],
                    lost=ev['lost'],common_remaining=ev['residual_shared_errors'],search_rows=ev['search_rows'],
                    expected_seconds=2*history[5]['worker_wall_seconds'],limit_seconds=cfg['trial_timeout_seconds']))
        result=dict(model=DEEP_MODEL,files=pins,reference_sha256=digest(args.reference/'protocol.json'),decisions=evidence,
                    requested=any(e['selected']==TEXT_DEEP for e in evidence))
        freeze_file(args.request,result)
        print('Requested by',sum(e['selected']==TEXT_DEEP for e in evidence),'of',len(evidence),'training replays',flush=True)
        return
    request=read(args.request)
    expect(request['requested'] and request['model']==DEEP_MODEL and request['files']==pins,'No matching frozen weight request')
    expect(any(e['selected']==TEXT_DEEP and e['expected_seconds']<=e['limit_seconds'] for e in request['decisions']),
           'No eligible training decision')
    def verified(name):
        _,sha,size=DEEP_FILES[name]
        file=args.weights/name
        return file.is_file() and file.stat().st_size==size and digest(file)==sha
    if args.record.exists():
        existing=read(args.record)
        expect(existing['request_sha256']==digest(args.request) and existing['pins']==pins and all(verified(n) for n in pins),
               'Existing download receipt or weights no longer match')
        print('Verified completed download',DEEP_MODEL,flush=True)
        return
    first=args.record.with_suffix('.initial.json')
    if first.exists():
        initial=read(first)
        expect(initial['request_sha256']==digest(args.request),'Unfinished request differs')
    else:
        initial=dict(request_sha256=digest(args.request),missing=[n for n in pins if not verified(n)])
        freeze_file(first,initial)
    fetched=acquire(args.weights.resolve(),DEEP_MODEL,DEEP_FILES)
    downloaded=[n for n in pins if n in initial['missing'] or n in fetched]
    for name in DEEP_FILES:
        expect(verified(name),'Public weight verification failed')
    freeze_file(args.record,dict(model=DEEP_MODEL,request_sha256=digest(args.request),pins=pins,downloaded=downloaded,
                                 downloaded_bytes=sum(DEEP_FILES[n][2] for n in downloaded),model_api_calls=0,api_rmb=0))
    print(DEEP_MODEL,downloaded,flush=True)


if __name__=='__main__':main()
