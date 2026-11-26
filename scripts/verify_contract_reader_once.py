"""One optional campus extraction on anonymous synthetic documentation, followed by an offline replay."""
import argparse
import io
import json
import os
from pathlib import Path
import sqlite3
import time
import zipfile

import numpy as np

from scripts.campus_direct import CAMPUS_IP, HOST, campus_resolution
from superagent.ab_flash import ENDPOINT, MODEL
from superagent.contracts import TaskSpec
from superagent.data import load_training_task
from superagent.events import RunLogger
from superagent.experiment_llm import ExperimentLLM
from superagent.task_budget import TaskBudgetAccount, TaskBudgetLedger


def verify(args):
    root=args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    ledger=TaskBudgetLedger(args.ledger)
    second=args.case=='packing'
    scope='documented-field-order-'+('packing' if second else 'check')+'-1126'
    ceiling=.015
    ledger.preflight(scope,ceiling,fresh_outbox_scope=True)
    loggers=[]
    for name in ('first','replay'):
        folder=root/name
        folder.mkdir()
        log=RunLogger(folder)
        log.initialize_usage_ledger()
        loggers.append(log)
    archive=root/'anonymous.zip'
    with zipfile.ZipFile(archive,'x') as package:
        note=('In every stored observation the following is the actual feature layout: mass, length, volume, count.'
              if second else 'Instrument coordinate layout follows the acquisition sequence: east, north, height.')
        width=4 if second else 3
        package.writestr('README.md',note+'\nReport accuracy.')
        for name,array in [('C_train.npy',np.arange(12*width).reshape(12,width)),('y_train.npy',np.tile([1,2,3],4) if second else np.tile([0,1],6))]:
            stream=io.BytesIO()
            np.save(stream,array)
            package.writestr(name,stream.getvalue())
    account=TaskBudgetAccount(ledger,scope,str(root),fresh_outbox_scope=True)
    client=ExperimentLLM(ENDPOINT,str(args.key_file),MODEL,ceiling,max_requests=1,allowed_served_models={MODEL},
                         account=account,streaming=True)
    client.set_deadline(time.monotonic()+120)
    os.environ['SUPERAGENT_CONTRACT_CACHE_DIR']=str(root/'cache')
    os.environ['NO_PROXY']=os.environ['no_proxy']=HOST
    with campus_resolution(CAMPUS_IP):
        first=load_training_task(archive,read_declarations=True,declaration_client=client,log=loggers[0])
    replay=load_training_task(archive,read_declarations=True,log=loggers[1])
    one,two=[TaskSpec.from_task(t).record() for t in (first,replay)]
    for log,contract in zip(loggers,(one,two)):
        log.write_json('task_contract.json',contract)
    claim=one['goal_resolution']['fields']['columns']
    journals=[[json.loads(line) for line in log.usage_path.read_text().splitlines()][1:] for log in loggers]
    with sqlite3.connect(args.ledger.resolve().as_uri()+'?mode=ro',uri=True) as db:
        receipts=db.execute('SELECT id,status FROM requests WHERE scope=?',(scope,)).fetchall()
    replay_events=[json.loads(line) for line in loggers[1].events_path.read_text().splitlines()]
    cached=any(e.get('kind')=='contract_column_evidence' and e.get('cache_hit') is True for e in replay_events)
    paid=journals[0]
    metered=(len(paid)==1 and receipts==[(paid[0].get('budget_request_id'),'known')]
             and all(type(paid[0].get(k)) is int for k in ('input_tokens','output_tokens')))
    result=dict(same_contract=one==two,status=claim['status'],fields=claim['value'],requests=client.request_count,
        replay_requests=len(journals[1]),cache_hit=cached,ledger_receipt_matched=metered,classifier_fits=0,
        usage=[{k:e.get(k) for k in ('served_model','input_tokens','output_tokens','course_cost_rmb')} for e in paid])
    loggers[0].write_json('verification.json',result)
    print(json.dumps(result,ensure_ascii=False))
    expected=['mass','length','volume','count'] if second else ['east','north','height']
    valid=(claim['status'],claim['value']) in [('proposed',expected),('unknown',None)]
    return int(one!=two or client.request_count!=1 or not metered or bool(journals[1]) or not cached or not valid)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('output','ledger','key-file'):
        p.add_argument('--'+k,required=True,type=Path)
    p.add_argument('--case',choices=('coordinates','packing'),default='coordinates')
    raise SystemExit(verify(p.parse_args()))
