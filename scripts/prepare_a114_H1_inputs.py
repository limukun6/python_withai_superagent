"""Prepare the two original A-registered H1 training regions; never score or export outer holdout."""
import argparse
import csv
import gzip
import io
import json
import os
from pathlib import Path
import time
from unittest.mock import patch

import pyarrow.parquet as pq

from scripts.a_batch_common import save
from scripts.materialize_b_c1_rows import stored_zip
from scripts.run_a10_real_task_pilot import read, sha
from scripts import run_b1_ab_acceptance as fixture
from superagent import ab_run
from superagent.contracts import TaskSpec
from superagent.data import load_training_task
from superagent.metrics import MetricSpec
from superagent.prediction_evidence import fingerprint
from superagent.splits import grouped_partition
from tools.ab_registered_rows import content_digest, registered_partitioner

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'assets', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    frozen = ab_run.sources()
    registry = ROOT/'research/a12_a16_day/a13_H1_publisher/registration.json'
    reg = read(registry)['H1']
    save(a.output/'protocol.json', dict(batch='A114', runner_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT/'docs/batches/A114_H1_scope_and_inputs_plan.md'), source_files=frozen,
         original_registration_sha256=sha(registry), pool_rows=2000, outer_seeds=list(reg['partitions']),
         inner_seeds=[707,808,909], inner_registration='new explicit training-only grouped partitions',
         actual_classifier_fits=0, real_API_requests=0, outer_performance_access_allowed=False))
    try:
        fixture.offline(a.output)
        if sha(a.source) != reg['source']['sha256']:
            raise ValueError('Original publisher archive differs')
        bref = ROOT/'research/b1_05_pair_pilot/h1'
        breg = read(bref/'results.json')
        raw = gzip.decompress((bref/'source_ids.json.gz').read_bytes())
        if fixture.sha(raw) != breg['source_ids_sha256'] or breg['source_sha256'] != reg['source']['sha256']:
            raise ValueError('B source/registration identity differs')
        bids = json.loads(raw)
        for seed, row in bids.items():
            if len(row['outer_train_source_ids']) != breg['partitions'][seed]['train_rows']:
                raise ValueError('B global training IDs differ')
            for key, value in row['inner'].items():
                if fixture.digest(value) != breg['partitions'][seed]['inner_row_ids_sha256'][key]:
                    raise ValueError('B global inner IDs differ')
        os.environ['SUPERAGENT_MINILM_WEIGHTS'] = str(a.assets/'minilm')
        results = {}
        for seed, outer in reg['partitions'].items():
            ids = [reg['selected_original_ids'][i] for i in outer['train']]
            held_ids = {reg['selected_original_ids'][i] for i in outer['held']}
            if len(ids) != len(set(ids)) or set(ids) & held_ids or len(ids)+len(held_ids) != 2000:
                raise ValueError('A sealed outer boundary differs')
            selected = pq.read_table(a.source, columns=['id','sentence1','sentence2','label'],
                                     filters=[('id','in',ids)]).to_pylist()
            lookup = {row['id']: row for row in selected}
            if set(lookup) != set(ids) or len(selected) != len(ids):
                raise ValueError('Filtered source rows differ')
            folder = a.output/seed
            folder.mkdir()
            table = io.StringIO(newline='')
            writer = csv.writer(table, lineterminator='\n')
            writer.writerow(['left','right','label'])
            for key in ids:
                row = lookup[key]
                writer.writerow([row['sentence1'],row['sentence2'],int(row['label'])])
            manifest = dict(schema_version=1,objective='classification',metric='accuracy',modality='pair_text',
                            pair_semantics='symmetric',pair_label_one='equivalent',target='label',
                            columns=[dict(name=v,type='text') for v in ('left','right')],
                            splits=dict(train=dict(format='csv',path='pairs.csv')))
            stored_zip(folder/'train.zip',[('manifest.json',json.dumps(manifest,sort_keys=True)),('pairs.csv',table.getvalue())])
            task = load_training_task(folder/'train.zip')
            if len(task.y_train) != len(ids) or task.spec['input_schema']['pair_semantics'] != 'symmetric':
                raise ValueError('Official import contract differs')
            splits = []
            for index, inner in enumerate((707,808,909)):
                fit, search, _ = grouped_partition(task, inner, 'shared_text')
                splits.append(dict(id=('primary','confirmation_1','confirmation_2')[index],seed=inner,
                                   train_indices=fit.tolist(),search_indices=search.tolist(),
                                   row_ids_sha256=fingerprint(search),label_sha256=fingerprint(task.y_train[search])))
            save(folder/'inner_splits.json',dict(version=1,kind=task.kind,training_content_sha256=content_digest(task.X_train,task.y_train),
                                               grouping='shared_text',splits=splits))
            # save() uses Windows text translation; canonicalize the newly-created file before registering it.
            signed = folder/'inner_splits.json'
            signed.write_bytes(signed.read_bytes().replace(b'\r\n',b'\n'))
            digest = sha(signed)
            selector = registered_partitioner(task,signed,digest)
            workers = folder/'worker_inputs'
            workers.mkdir()
            with patch.object(ab_run,'grouped_partition',selector):
                _, actual = ab_run.freeze_partitions(task,workers,707,MetricSpec.for_task(task))
            spec = TaskSpec.from_task(task).record()
            caps = ab_run.registered_capabilities(spec,len(ids),'full')
            if len(caps) != 8 or any(c['swap_label_rule'] != 'preserves' for c in caps.values()):
                raise ValueError('Expected eight safe symmetric routes')
            results[seed] = dict(train_rows=len(ids),outer_holdout_rows=len(held_ids),
                                train_source_ids_sha256=fixture.digest(ids),train_archive_sha256=sha(folder/'train.zip'),
                                inner_manifest_sha256=digest,registered_splits=actual,task_spec=spec,capabilities=caps)
            print(dict(seed=seed,training_rows=len(ids),safe_symmetric_candidates=len(caps)),flush=True)
        if frozen != ab_run.sources():
            raise ValueError('Production source changed')
        save(a.output/'results.json',dict(training_inputs=results,B_full_registration_metadata_checked=True,
             B_full_rows=breg['rows'],B_full_outer_rows={s:r['train_rows'] for s,r in breg['partitions'].items()},
             H1_final_scope_frozen=False,outer_rows_exported=0,outer_scores_computed=0))
        save(a.output/'completion.json',dict(status='training_inputs_prepared_pending_joint_scope_freeze',
             outer_training_packages=2,inner_partitions=6,actual_classifier_fits=0,real_API_requests=0,
             model_predictions=0,outer_scores_computed=0,source_unchanged=True,elapsed_seconds=time.monotonic()-started))
    except Exception as error:
        save(a.output/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,
             real_API_requests=0,elapsed_seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
