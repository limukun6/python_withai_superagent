"""Check A's registered inputs without H evaluation and run one real development path."""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import zipfile

import polars as pl

from scripts import run_b1_ab_acceptance as records
from scripts import run_b1_05_pair_pilot as pilot
from scripts import run_b1_06_mrpc as mrpc
from superagent.ab_catalog import capabilities_for
from superagent.ab_run import sources
from superagent.ab_sentence import verified_directory
from superagent.ab_symmetric import CHANNELS, ENCODER
from superagent.contracts import TaskSpec
from superagent.data import load_development_task


METHODS = [*CHANNELS, ENCODER]
PLAN = records.ROOT / 'docs/batches/BA12_equivalence_plan.md'


def qualify_paws(source, destination):
    archive = records.ROOT / 'research/a12_a16_day'
    ref = records.read(archive / 'a13_H1_publisher/registration.json')['H1']
    shared = records.read(archive / 'a13_holdout_packages/registration.json')['H1']
    if records.sha(source.read_bytes()) != ref['source']['sha256'] or ref['partitions'] != shared['partitions']:
        raise ValueError('Published source or A partition identity differs')
    frame = pl.read_parquet(source, columns=['id','sentence1','sentence2','label'])
    positions = ref['selected_source_positions']
    rows = [frame.row(i) for i in positions]
    if len(set(positions)) != len(rows) or [r[0] for r in rows] != ref['selected_original_ids']:
        raise ValueError('Original example identities changed')
    raw = io.StringIO(newline='')
    writer = csv.writer(raw, delimiter='\t')
    writer.writerow(['id','sentence1','sentence2','label'])
    writer.writerows(rows)
    if records.sha(raw.getvalue().encode()) != ref['files']['train.tsv']:
        raise ValueError('Registered 2,000-row training TSV differs')
    output = {}
    for seed, block in ref['partitions'].items():
        fit, held = block['train'], block['held']
        if sorted(fit + held) != list(range(len(rows))):
            raise ValueError('Partition does not cover the registered rows exactly once')
        first = {text for i in fit for text in rows[i][1:3]}
        if first.intersection(text for i in held for text in rows[i][1:3]):
            raise ValueError('A source sentence crosses the registered outer boundary')
        table = io.StringIO(newline='')
        writer = csv.writer(table)
        writer.writerow(['sentence1','sentence2','label'])
        writer.writerows(rows[i][1:] for i in fit)
        spec = shared['task_spec']
        manifest = dict(schema_version=1, objective='classification', metric='accuracy',
                        **spec['input_schema'], splits={'train':{'format':'csv','path':'train.csv'}})
        packet = destination / ('training_' + seed + '.zip')
        with zipfile.ZipFile(packet, 'x') as out:
            pilot.write_member(out,'manifest.json',json.dumps(manifest,sort_keys=True).encode())
            pilot.write_member(out,'train.csv',table.getvalue().encode())
        task = load_development_task(packet)
        contract = TaskSpec.from_task(task).record()
        caps = capabilities_for(contract, train_rows=len(task.y_train))
        if (contract != spec or task.X_val is not None or task.y_val is not None or
                list(caps) != METHODS or task.y_train.tolist() != [rows[i][3] for i in fit]):
            raise ValueError('Shared equivalence contract, labels or legal methods changed')
        output[seed] = dict(training=len(fit),held=len(held),fit_ids_sha256=records.digest(fit),
                            held_ids_sha256=records.digest(held),training_packet_sha256=records.sha(packet.read_bytes()),
                            methods=list(caps),catalog_sha256=records.digest(caps))
    return dict(source_sha256=ref['source']['sha256'], registered_tsv_sha256=ref['files']['train.tsv'],
                registered_rows=len(rows),partitions=output,fits=0,predictions=0,scores=0)


def run(args):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    records.offline(root)
    os.environ.update(SUPERAGENT_OFFLINE='1',SUPERAGENT_MINILM_WEIGHTS=str(args.weights.resolve()),
                      OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    try:
        if verified_directory() is None:
            raise ValueError('Registered, local MiniLM is required for this four-method check')
        h1 = qualify_paws(args.paws,root)
        x,y,ids = mrpc.load_rows(args.mrpc)
        old = records.read(records.ROOT/'research/b1_06_mrpc/protocol.json')['tasks']['45101_extended']
        if records.digest(ids) != old['source']['published_ids_in_source_order_sha256']:
            raise ValueError('MRPC publication order changed')
        task = pilot.prepare_pairs(root,'mrpc',x,y,'symmetric',old['source'],outer_seed=45101)
        if (task['row_ids_sha256'] != old['row_ids_sha256'] or task['candidate_ids'] != METHODS or
                task['inner_input_rows'] != old['inner_input_rows'] or task['outer_rows'] != old['outer_rows']):
            raise ValueError('MRPC rows or registered candidate identities changed')
        protocol = dict(batch='BA12-equivalence',source=records.digest(sources()),runtime=pilot.environment(),
                        code=records.sha(Path(__file__).read_bytes()),plan=records.sha(PLAN.read_bytes()),
                        h1=h1,mrpc=task,fit_limit=11,model_api_limit=0)
        records.write(root/'protocol.json',protocol)
        if args.reference is None:
            records.write(root/'completion.json',dict(status='registered',fits=0,model_api_requests=0,h_scores=0))
            return
        if records.read(args.reference/'protocol.json') != protocol:
            raise ValueError('Actual run differs from the registered protocol')
        path = root/'mrpc'
        outcome = pilot.train_one(path,task,protocol['source'])
        if len(records.trace(root)) != outcome['fits']:
            raise ValueError('Actual supervised fit counters disagree')
        records.write(root/'all_models_frozen.json',{'mrpc':outcome['model_sha256']})
        outcome['delivery'] = pilot.score_one(path,task)
        records.write(root/'results.json',dict(h1=h1,mrpc=outcome))
        records.write(root/'completion.json',dict(status='passed',development_paths=1,fits=outcome['fits'],
                      model_api_requests=0,h_fits=0,h_predictions=0,h_scores=0,course_test_accessed=False))
    except Exception as error:
        records.write(root/'completion.json',dict(status='failed',reason=str(error),fits=len(records.trace(root)),
                      model_api_requests=0,h_scores=0,automatic_retry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('mrpc','paws','weights','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--reference',type=Path)
    run(parser.parse_args())
