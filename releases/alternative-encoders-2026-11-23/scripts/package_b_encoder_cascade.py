"""Archive the early-exit receipts and bind previously published native predictions to the new controls."""
import argparse
from pathlib import Path

import numpy as np

from scripts.package_b_grouped_titles import copy_receipt
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_encoder_vote_followup import classes
from scripts.run_b_local_component_pair import check, read
from superagent.ab_methods import TRANSFER_METHODS
from superagent.ab_run import sha
from superagent.prediction_evidence import fingerprint


def copy_batch(source, target):
    end = read(source/'completion.json')
    for name in ('decisions','exports','protocol','results'):
        path=source/(name+'.json')
        check(sha(path)==end[name],'Unbound '+name)
        copy_receipt(path,target/path.name)
    copy_receipt(source/'completion.json',target/'completion.json')
    for name,seal in end['groups'].items():
        f=source/name/'grouped_correctness.json'
        check(sha(f)==seal,'Unbound grouped counts')
        copy_receipt(f,target/name/f.name)
    for pattern in ('fit_started.json','fit_completed.json'):
        for f in source.rglob(pattern):
            copy_receipt(f,target/f.relative_to(source))


def run(a):
    decisions=read(a.initial/'decisions.json')
    refs=read(a.initial/'protocol.json')['tasks']
    check(decisions['films']['policy']['first']==TRANSFER_METHODS['deep-e5'][1],'Unexpected first movie model')
    title_seal=read(a.titles/'exports_frozen.json')
    q_seal=read(a.films/'outer_export.json')
    check(title_seal['model'] in {r['model'] for r in refs['titles']['model_refs'].values()} and
          q_seal['model']==refs['films']['model_refs'][decisions['films']['policy']['first']]['model'],'Native models differ')
    inventory={}
    exports=read(a.initial/'exports.json')
    for name,native,seal,arm in [('titles',a.titles,title_seal['development'],'full'),
                                 ('films',a.films,q_seal['predictions'],'first')]:
        output=a.initial/name/'predictions.npz'
        check(sha(output)==exports[name]['file'],'Different routed model output')
        with np.load(output) as data:
            vec=data[arm]
        for repeat in (0,1):
            f=native/('autonomous_export_'+str(repeat))/'rows.csv'
            np.testing.assert_array_equal(classes(f,seal,len(vec)),vec)
        inventory[name]=dict(column=arm,rows=len(vec),reference_csv=seal,repeated_native_exports=2,vector=fingerprint(vec))
    copy_batch(a.initial,a.out/'initial')
    frozen_save(a.out/'initial/native_controls.json',inventory)
    copy_batch(a.replication,a.out/'replication')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('initial','replication','titles','films','out'):
        p.add_argument('--'+n,type=lambda x:Path(x).resolve(),required=True)
    run(p.parse_args())
