"""Resolve trusted, train-local split IDs and check them against the current task contents."""
import hashlib
import json
from pathlib import Path

import numpy as np

from superagent.prediction_evidence import fingerprint
from superagent.splits import effective_grouping, sample_groups


def content_digest(rows, labels):
    digest = hashlib.sha256()
    for row, label in zip(rows, labels, strict=True):
        value = row.tolist() if isinstance(row, np.ndarray) else row
        scalar = label.item() if isinstance(label, np.generic) else label
        digest.update(json.dumps([value, scalar], separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode() + b'\n')
    return digest.hexdigest()


def validated_ids(record, labels, groups):
    fit, held = record['train_indices'], record['search_indices']
    if not isinstance(fit, list) or not isinstance(held, list) or not fit or not held:
        raise ValueError('Both sides require nonempty lists of row IDs')
    all_ids = fit + held
    if any(type(i) is not int for i in all_ids) or sorted(all_ids) != list(range(len(labels))):
        raise ValueError('Partition IDs must cover the current input exactly once')
    if set(groups[fit]) & set(groups[held]):
        raise ValueError('An input group crosses the registered boundary')
    domain = set(labels.tolist())
    if set(labels[fit].tolist()) != domain or set(labels[held].tolist()) != domain:
        raise ValueError('The registered partition lost a class')
    if (fingerprint(held) != record['row_ids_sha256'] or
            fingerprint(labels[held]) != record['label_sha256']):
        raise ValueError('Registered row order or held labels changed')
    return np.asarray(fit, dtype=int), np.asarray(held, dtype=int)


def registered_partitioner(task, manifest_file, expected_sha256):
    """The returned callable has the same signature as the grouped split function used by A."""
    raw = Path(manifest_file).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Inner partition manifest differs from the registered digest')
    manifest = json.loads(raw)
    if (manifest['version'] != 1 or manifest['kind'] != task.kind or
            manifest['training_content_sha256'] != content_digest(task.X_train, task.y_train)):
        raise ValueError('Training content or its order differs from the registered partition manifest')
    mode = effective_grouping(task.kind, 'shared_text')
    if manifest['grouping'] != mode:
        raise ValueError('Partition manifest has the wrong group policy')
    groups = sample_groups(task, mode)
    registered, held_sets = {}, set()
    for row in manifest['splits']:
        seed = row['seed']
        if type(seed) is not int or seed in registered:
            raise ValueError('Ambiguous registered split seed')
        train, held = validated_ids(row, task.y_train, groups)
        held_set = frozenset(held.tolist())
        if held_set in held_sets:
            raise ValueError('A named partition reuses held rows')
        held_sets.add(held_set)
        registered[seed] = train, held, dict(total_groups=len(set(groups)), train_groups=len(set(groups[train])),
            heldout_groups=len(set(groups[held])), class_domain_preserved=True, selection_rule='pre_registered_source_rows')

    def partition(current_task, seed, grouping='exact_rows'):
        if current_task is not task or effective_grouping(task.kind, grouping) != mode or seed not in registered:
            raise ValueError('Task, grouping or seed differs from the registered split request')
        train, held, summary = registered[seed]
        return train.copy(), held.copy(), dict(summary)

    return partition
