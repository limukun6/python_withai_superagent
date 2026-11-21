"""Explicitly reviewed publisher train-only PAWS Parquet source; no model evaluation."""
import argparse
import json
from pathlib import Path
import sys
import time

from scripts.a_batch_common import save
from scripts import prepare_a13_photo_and_holdouts as common


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    parser.add_argument('--preparation-dependencies',type=Path,required=True)
    args=parser.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    save(root/'protocol.json',dict(batch='A13-H1-publisher',runner_sha256=common.digest(Path(__file__).read_bytes()),
        plan_sha256=common.digest((Path(__file__).resolve().parents[1]/'docs/batches/A13_photo_and_H_registration_plan.md').read_bytes()),
        real_api_requests=0,model_fits=0,official_validation_test_downloaded=False))
    try:
        common.PROXY='http://127.0.0.1:7897'
        expected='8dc9ad3e5f30ad9a86b290fe236d528ef23a5751fec9a35d99cbacf68ba277cf'
        record=common.download('https://huggingface.co/datasets/google-research-datasets/paws/resolve/main/labeled_final/train-00000-of-00001.parquet',
            root/'train.parquet',expected)
        sys.path.insert(0,str(args.preparation_dependencies.resolve()))
        import pyarrow
        import pyarrow.parquet as pq
        rows=pq.read_table(root/'train.parquet').to_pylist()
        if len(rows)!=49401:raise ValueError('Unexpected publisher PAWS train row count')
        result=common.register_pair_rows(root,rows,record,['train.parquet'])
        result['readiness']='symmetric label parser implemented; safe B methods still pending'
        result['parquet_preparation_version']=pyarrow.__version__
        result['source_equivalence']='Publisher train source; unavailable original tar not compared byte-for-byte'
        save(root/'registration.json',dict(H1=result))
        save(root/'completion.json',dict(status='prepared',actual_classifier_fits=0,real_api_requests=0,
            holdout_scores_opened=False,elapsed_seconds=time.monotonic()-began))
        print(json.dumps(dict(rows=result['rows'],status='registered',model_fits=0)))
    except Exception as error:
        save(root/'completion.json',dict(status='failed',error=str(error),actual_classifier_fits=0,
            real_api_requests=0,elapsed_seconds=time.monotonic()-began));raise
