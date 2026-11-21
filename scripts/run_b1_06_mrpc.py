"""Two registered outer splits, paired local resource arms, with all models frozen before scoring."""
import argparse
import csv
import gzip
import json
import os
from pathlib import Path
import shutil
import time
from unittest.mock import patch

import joblib
import numpy as np

from scripts import run_b1_ab_acceptance as fixture
from scripts import run_b1_05_pair_pilot as pilot
from superagent.ab_catalog import capabilities_for
from superagent.ab_run import sources
from superagent.ab_sentence import verified_directory
from superagent.ab_symmetric import CHANNELS, ENCODER
from superagent.contracts import TaskSpec
from superagent.data import load_development_task


SHA = '61fd41301e0e244b0420c4350a170c8e7cf64740335fc875a4af2d79af0df0af'
REVISION = 'a3f3ab759b9d7018046ce361f115ef38276d2e1e'
ARMS = ('lexical', 'extended')
ORDER = ((45101, 'lexical'), (45101, 'extended'), (45102, 'extended'), (45102, 'lexical'))


def load_rows(source):
    import polars as pl
    if fixture.sha(source.read_bytes()) != SHA:
        raise ValueError('MRPC train SHA256 differs from the registered publication')
    frame = pl.read_parquet(source, columns=['idx', 'sentence1', 'sentence2', 'label'])
    if len(frame) != 3668 or any(frame.null_count().row(0)) or frame['idx'].n_unique() != 3668:
        raise ValueError('Unexpected MRPC source rows')
    x = list(zip(frame['sentence1'].to_list(), frame['sentence2'].to_list(), strict=True))
    y = np.asarray(frame['label'].to_list(), dtype=str)
    return x, y, frame['idx'].to_list()


def weights_for(arm, weights):
    return patch.dict(os.environ, {'SUPERAGENT_MINILM_WEIGHTS': str(weights) if arm == 'extended' else ''})


def prepare(root, source, weights):
    x, y, source_ids = load_rows(source)
    pilot.identities_file(root / 'source_id_order.json.gz', source_ids)
    metadata = dict(dataset='GLUE/MRPC', published_split='train', source_sha256=SHA, revision=REVISION,
                    published_ids_in_source_order_sha256=fixture.digest(source_ids))
    entries = {}
    for seed in (45101, 45102):
        with weights_for('lexical', weights):
            entry = pilot.prepare_pairs(root, str(seed), x, y, 'symmetric', metadata, outer_seed=seed)
        shared = root / str(seed)
        task = load_development_task(shared / 'train.zip')
        spec = TaskSpec.from_task(task).record()
        modes = {}
        for arm in ARMS:
            with weights_for(arm, weights):
                caps = capabilities_for(spec, train_rows=len(task.y_train))
            modes[arm] = caps
            folder = f'{seed}_{arm}'
            shutil.copytree(shared, root / folder)
            entries[folder] = dict(entry, name=folder, outer_seed=seed, resources=arm,
                                   catalog_sha256=fixture.digest(caps), candidate_ids=list(caps))
        if list(modes['lexical']) != list(CHANNELS) or list(modes['extended']) != [*modes['lexical'], ENCODER]:
            raise ValueError('The two catalogues must differ only by the registered optional candidate')
        if any(modes['lexical'][name] != modes['extended'][name] for name in modes['lexical']):
            raise ValueError('Shared capabilities differ between resource conditions')
    return entries


def paired_counts(root, seed):
    def predictions(arm):
        with (root / f'{seed}_{arm}' / 'pred_0/rows.csv').open(newline='') as stream:
            return [r['prediction'] for r in csv.DictReader(stream)]
    a, b = predictions('lexical'), predictions('extended')
    truth = joblib.load(root / str(seed) / 'holdout.joblib')['y']
    if len(a) != len(truth) or len(b) != len(truth):
        raise ValueError('Paired prediction lengths disagree')
    left, right = np.asarray(a) == truth, np.asarray(b) == truth
    return dict(rows=len(a), lexical_correct=int(left.sum()), extended_correct=int(right.sum()),
                newly_correct=int((~left & right).sum()), newly_wrong=int((left & ~right).sum()),
                both_correct=int((left & right).sum()), both_wrong=int((~left & ~right).sum()),
                difference_percentage_points=float(100 * (right.mean() - left.mean())))


def run(root, source, weights):
    root.mkdir(parents=True, exist_ok=False)
    fixture.offline(root)
    os.environ.update(SUPERAGENT_OFFLINE='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2')
    started = time.monotonic()
    runs = []
    try:
        with weights_for('extended', weights):
            if verified_directory() is None:
                raise ValueError('Pinned MiniLM files or required local runtime unavailable')
            environment = pilot.environment()
        frozen = fixture.digest(sources())
        entries = prepare(root, source, weights)
        fixture.write(root / 'protocol.json', dict(batch='B1-06', source_sha256=frozen, environment=environment,
            driver_sha256=fixture.sha(Path(__file__).read_bytes()), helper_sha256=fixture.sha(Path(pilot.__file__).read_bytes()),
            plan_sha256=fixture.sha((pilot.ROOT / 'docs/batches/B1_06_plan.md').read_bytes()), source_file_sha256=SHA,
            source_revision=REVISION, execution_order=ORDER, tasks=entries, fit_limit_per_path=11, fit_limit_batch=44,
            api_requests_allowed=0, inner_seed=pilot.INNER_SEED, wall_seconds=pilot.WALL, trial_seconds=pilot.TRIAL))
        for seed, arm in ORDER:
            name = f'{seed}_{arm}'
            before = len(fixture.trace(root))
            with weights_for(arm, weights):
                row = pilot.train_one(root / name, entries[name], frozen)
            if len(fixture.trace(root)) - before != row['fits']:
                raise ValueError('Independent fit count differs on ' + name)
            runs.append(row)
            fixture.write(root / 'results.json', dict(paths=runs))
        fixture.write(root / 'all_models_frozen.json', {r['name']: r['model_sha256'] for r in runs})
        for row in runs:
            name = row['name']
            with weights_for(entries[name]['resources'], weights):
                row['delivery'] = pilot.score_one(root / name, entries[name])
        pairs = {str(seed): paired_counts(root, seed) for seed in (45101, 45102)}
        held = [set(json.loads(gzip.decompress((root / str(seed) / 'row_ids.json.gz').read_bytes()))['outer']['held'])
                for seed in (45101, 45102)]
        if frozen != fixture.digest(sources()):
            raise ValueError('Production code changed during the registered batch')
        fixture.write(root / 'results.json', dict(paths=runs, paired_counts=pairs, outer_holdout_shared_rows=len(held[0] & held[1])))
        fixture.write(root / 'completion.json', dict(status='passed', fits=len(fixture.trace(root)), paths=len(runs),
                      api_requests=0, api_rmb=0, models_frozen_before_scoring=True, h1_accessed=False,
                      library_validation_and_test_opened=False, seconds=time.monotonic() - started))
    except Exception as error:
        fixture.write(root / 'results.json', dict(paths=runs))
        fixture.write(root / 'completion.json', dict(status='failed', error=str(error), fits_observed=len(fixture.trace(root)),
                      api_requests=0, automatic_retry=False))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-parquet', required=True, type=Path)
    parser.add_argument('--weights', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    run(args.output.resolve(), args.training_parquet.resolve(), args.weights.resolve())
