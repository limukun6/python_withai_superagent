"""Audit existing sealed H2 source identities only; no images, models or network."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    def offline(event,args):
        if event in ('socket.connect','socket.getaddrinfo','socket.sendto'):raise PermissionError('A104 no network')
    sys.addaudithook(offline)
    folder=ROOT/'research/bh02_image_seal';seal=read(folder/'results.json')
    files=[folder/n for n in ('source_ids.json.gz','results.json','protocol.json')]
    files += [ROOT/'research/bh_03_training_inputs'/n for n in ('results.json','protocol.json','completion.json')]
    save(a.output/'protocol.json',dict(batch='A104',runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A104_H2_registration_audit_plan.md'),
        input_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in files},
        image_access_allowed=False,actual_classifier_fits=0,real_API_requests=0))
    try:
        raw=gzip.decompress((folder/'source_ids.json.gz').read_bytes())
        if hashlib.sha256(raw).hexdigest()!=seal['source_ids_sha256']:raise ValueError('Original seal source identities changed')
        records=json.loads(raw);prepared=read(ROOT/'research/bh_03_training_inputs/results.json');all_ids=set();rows={}
        if set(records)!={'45101','45102'}:raise ValueError('Outer registration differs')
        for seed,row in records.items():
            train=row['outer_train_source_ids'];held=row['outer_holdout_source_ids']
            if len(train)!=len(set(train)) or len(held)!=len(set(held)) or len(train)!=2500 or len(held)!=500:
                raise ValueError('Invalid outer row counts')
            if set(train)&set(held) or all_ids&(set(train)|set(held)):raise ValueError('Outer identity overlap')
            all_ids.update(train);all_ids.update(held);splits={}
            for inner,part in row['inner'].items():
                fit=part['train_source_ids'];query=part['search_source_ids']
                if len(fit)!=2000 or len(query)!=500 or set(fit)&set(query) or set(fit)|set(query)!=set(train):
                    raise ValueError('Inner identities escape the registered training region')
                splits[inner]=dict(train=2000,search=500)
            if set(splits)!={'707','808','909'}:raise ValueError('Inner seeds differ')
            rows[seed]=dict(training_rows=2500,held_region_rows=500,inner_splits=splits,
                prepared_training_archive_sha256=prepared[seed]['train_archive_sha256'],
                prepared_inner_manifest_sha256=prepared[seed]['inner_manifest_sha256'])
        save(a.output/'results.json',dict(outer=rows,unique_source_ids=len(all_ids),
            original_RGB_archive_sha256=seal['archive_sha256'],
            reusable='B sealed IDs, official-image decoder builder, registered-partition manifests and previous validation receipts',
            local_preparation_remaining='Verify/acquire exact RGB archive and rebuild train-only inputs; no claim they exist locally',
            metadata_visibility='Source names and split memberships read; image pixels and model scores not read',
            formal_H_strategy_frozen=False))
        save(a.output/'completion.json',dict(status='passed',outer_registrations=2,inner_registrations=6,
            unique_source_ids=len(all_ids),image_members_opened=0,model_predictions=0,
            actual_classifier_fits=0,real_API_requests=0,elapsed_seconds=time.monotonic()-start))
        print(dict(outer=2,inner=6,unique_ids=len(all_ids),image_members_opened=0))
    except Exception as e:
        save(a.output/'completion.json',dict(status='stopped',error=str(e),actual_classifier_fits=0,real_API_requests=0));raise
