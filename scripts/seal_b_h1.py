"""Register PAWS-Wiki training rows for the future H1 transfer study without fitting or scoring."""
import argparse
from collections import Counter
from importlib.metadata import version
from pathlib import Path
import platform

import numpy as np

from scripts import run_b1_ab_acceptance as fixture
from scripts.run_b1_05_pair_pilot import identities_file
from superagent.data import TrainingTask
from superagent.splits import grouped_partition, overlap_audit, take


REVISION = '161ece9501cf0a11f3e48bd356eaa82de46d6a09'
SHA256 = '8dc9ad3e5f30ad9a86b290fe236d528ef23a5751fec9a35d99cbacf68ba277cf'
MEMBER = 'labeled_final/train-00000-of-00001.parquet'
SOURCE = 'https://huggingface.co/datasets/google-research-datasets/paws'


def aligned_split(task, seed):
    train, held, resource = grouped_partition(task, seed, 'shared_text')
    if overlap_audit(task, train, held)['search_rows_sharing_text']:
        raise ValueError('Shared text across a registered partition')
    return train, held, resource


def run(source, output):
    import polars as pl
    output.mkdir(parents=True, exist_ok=False)
    fixture.write(output / 'protocol.json', dict(dataset='PAWS-Wiki Labeled (Final)', author='Google Research',
                  source=SOURCE + '/resolve/' + REVISION + '/' + MEMBER, revision=REVISION,
                  required_file_sha256=SHA256, licence='https://github.com/google-research-datasets/paws/blob/master/LICENSE',
                  paper='https://aclanthology.org/N19-1131/', task='binary symmetric paraphrase identification',
                  class_meaning={'0': 'different meaning', '1': 'paraphrase'}, primary_metric='accuracy',
                  required_rows=49401, partition_source='published training split exclusively',
                  outer_seeds=[45101, 45102], inner_seeds=[707, 808, 909], grouping='shared_text',
                  python=platform.python_version(), packages={n: version(n) for n in ('numpy', 'scikit-learn', 'polars')},
                  group_code_sha256=fixture.sha((fixture.ROOT / 'superagent/splits.py').read_bytes()),
                  driver_sha256=fixture.sha(Path(__file__).read_bytes()), fit_limit=0, scoring=False))
    try:
        if fixture.sha(source.read_bytes()) != SHA256:
            raise ValueError('Source differs from the published SHA256')
        frame = pl.read_parquet(source, columns=['id', 'sentence1', 'sentence2', 'label'])
        if len(frame) != 49401 or frame['id'].n_unique() != len(frame) or any(frame.null_count().row(0)):
            raise ValueError('Unexpected published row identities or missing values')
        ids = frame['id'].to_numpy()
        pairs = list(zip(frame['sentence1'].to_list(), frame['sentence2'].to_list(), strict=True))
        labels = np.asarray(frame['label'].to_list(), dtype=str)
        if set(labels.tolist()) != {'0', '1'}:
            raise ValueError('Unexpected PAWS binary label encoding')
        full = TrainingTask('pair_text', pairs, labels)
        summary, all_ids, used = {}, {}, set()
        for outer in (45101, 45102):
            train, held, resource = aligned_split(full, outer)
            outer_held = frozenset(ids[held].tolist())
            if outer_held in used:
                raise ValueError('Outer partitions coincide')
            used.add(outer_held)
            sub = TrainingTask('pair_text', take(pairs, train), labels[train])
            rows = dict(outer_train_source_ids=ids[train].tolist(), outer_holdout_source_ids=ids[held].tolist(), inner={})
            inner_records = {}
            for inner in (707, 808, 909):
                fit, validation, _ = aligned_split(sub, inner)
                rows['inner'][str(inner)] = dict(train_source_ids=ids[train[fit]].tolist(),
                                                  search_source_ids=ids[train[validation]].tolist())
                inner_records[str(inner)] = fixture.digest(rows['inner'][str(inner)])
            if len(set(inner_records.values())) != 3:
                raise ValueError('Inner split row identities coincide')
            all_ids[str(outer)] = rows
            summary[str(outer)] = dict(train_rows=len(train), outer_holdout_rows=len(held), groups=resource['total_groups'],
                held_source_ids_sha256=fixture.digest(rows['outer_holdout_source_ids']), inner_row_ids_sha256=inner_records)
        rows_sha = identities_file(output / 'source_ids.json.gz', all_ids)
        fixture.write(output / 'results.json', dict(source_sha256=SHA256, rows=len(ids),
                      class_counts=dict(Counter(labels.tolist())), partitions=summary, source_ids_sha256=rows_sha))
        fixture.write(output / 'completion.json', dict(status='sealed', outer_partitions=2, inner_partitions_per_outer=3,
                      fits=0, predictions=0, performance_scores=0, api_requests=0, published_dev_test_opened=False))
    except Exception as error:
        fixture.write(output / 'completion.json', dict(status='failed', error=str(error), fits=0, performance_scores=0))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('training_parquet', type=Path)
    parser.add_argument('new_directory', type=Path)
    args = parser.parse_args()
    run(args.training_parquet, args.new_directory)
