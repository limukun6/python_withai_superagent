"""Offline H2 input reconstruction with explicit LF serialization and training-member guard."""
import argparse
import gzip
import json
import os
from pathlib import Path
import time
from unittest.mock import patch
import zipfile

from scripts.a_batch_common import hashes, save
from scripts.prepare_a105_H2_training_inputs import ROOT, EXPECTED
from scripts.run_a10_real_task_pilot import read, sha
from scripts import prepare_b_h2_training_inputs as builder
from scripts.seal_b_h2 import source_hashes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'assets', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    before = hashes(ROOT)
    references = ['scripts/prepare_b_h2_training_inputs.py', 'research/bh02_image_seal/source_ids.json.gz',
                  'research/bh_03_training_inputs/results.json']
    save(a.output/'protocol.json', dict(batch='A109', source_files=before,
         runner_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT/'docs/batches/A109_H2_canonical_inputs_plan.md'),
         reference_sha256={r: sha(ROOT/r) for r in references},
         original_source_sha256=EXPECTED, serialization='UTF-8, LF, same JSON formatting as B',
         actual_classifier_fits=0, real_API_requests=0, holdout_image_access_allowed=False))
    accesses = []
    try:
        builder.fixture.offline(a.output)
        if source_hashes(a.source) != EXPECTED:
            raise ValueError('Source SHA/author MD5 differs')
        records = json.loads(gzip.decompress((ROOT/references[1]).read_bytes()))
        allowed = {n for r in records.values() for n in r['outer_train_source_ids']}
        held = {n for r in records.values() for n in r['outer_holdout_source_ids']}
        if len(allowed) != 5000 or allowed & held:
            raise ValueError('Invalid sealed membership')
        original_open = zipfile.ZipFile.open

        def guarded_open(archive, name, mode='r', *args, **kwargs):
            if Path(archive.filename).resolve() == a.source.resolve():
                member = name.filename if isinstance(name, zipfile.ZipInfo) else name
                short = Path(member).name
                if mode != 'r' or short not in allowed:
                    raise ValueError('Attempt to access a non-training original image')
                accesses.append(short)
            return original_open(archive, name, mode, *args, **kwargs)

        def lf_write(path, value):
            Path(path).write_bytes((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8'))

        with patch.object(zipfile.ZipFile, 'open', guarded_open), patch.object(builder.fixture, 'write', lf_write):
            builder.prepare(a.source, a.output/'training')
        actual = read(a.output/'training/results.json')
        expected = read(ROOT/references[2])
        if set(actual) != set(expected):
            raise ValueError('Outer registrations differ')
        menus = {}
        os.environ['SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS'] = str(a.assets/'mobilenet_v3_small-047dcff4.pth')
        from superagent.ab_run import registered_capabilities
        from superagent.contracts import TaskSpec
        from superagent.data import load_training_task
        from tools.ab_registered_rows import registered_partitioner
        for seed, row in actual.items():
            if row != expected[seed]:
                raise ValueError('B registration mismatch: '+seed)
            task = load_training_task(a.output/'training'/seed/'train.zip')
            selector = registered_partitioner(task, a.output/'training'/seed/'inner_splits.json', row['inner_manifest_sha256'])
            for inner_seed in (707, 808, 909):
                selector(task, inner_seed)
            spec = TaskSpec.from_task(task).record()
            caps = registered_capabilities(spec, len(task.y_train), 'full')
            if len(caps) != 9 or 'image_mobilenet_v3_small_linear' not in caps:
                raise ValueError('Expected full9 image methods unavailable')
            menus[seed] = dict(rows=len(task.y_train), task_spec=spec, capabilities=caps)
        if len(accesses) != 5000 or set(accesses) != allowed or before != hashes(ROOT):
            raise ValueError('Training access or production source audit differs')
        save(a.output/'results.json', dict(registered_inputs=actual, menus=menus, source_unchanged=True,
             original_member_reads=len(accesses), unique_training_members=len(set(accesses)),
             original_source_sha256=EXPECTED, registration_exact_match=True,
             outer_holdout_image_members_opened=0))
        save(a.output/'completion.json', dict(status='prepared', training_photos_verified=5000,
             inner_partitions_verified=6, training_archives_match_B=True, inner_manifests_match_B=True,
             worker_payloads_match_B=True, actual_classifier_fits=0, real_API_requests=0,
             model_predictions=0, holdout_image_members_opened=0, formal_H_strategy_frozen=False,
             elapsed_seconds=time.monotonic()-start))
        print('Prepared H2: exact B training archives, manifests and six worker payloads; full9 menus; zero fits/API', flush=True)
    except Exception as error:
        save(a.output/'completion.json', dict(status='stopped', error=str(error),
             original_member_reads=len(accesses), actual_classifier_fits=0, real_API_requests=0,
             elapsed_seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    main()
