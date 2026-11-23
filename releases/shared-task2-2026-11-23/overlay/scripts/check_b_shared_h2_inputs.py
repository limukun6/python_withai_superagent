"""Verify already registered H2 training packs with the merged production reader, without training."""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np

from superagent import ab_vision
from superagent.ab_run import registered_capabilities, sha, sources
from superagent.contracts import TaskSpec
from superagent.data import load_development_task
from tools.ab_registered_rows import registered_partitioner

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'docs/batches/B_h2_receiver_preflight_20261123_zh.md'


def require(value, why):
    if not value:
        raise ValueError(why)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def offline(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
        raise PermissionError('Local training input check is offline')


def verify(args):
    sys.addaudithook(offline)
    os.environ[ab_vision.WEIGHT_ENV] = str(args.weights.resolve())
    production = sources()
    expected = read(ROOT / 'research/a126_merged_checks/merge_verification.json')['source_files']
    require({'superagent/' + k: v for k, v in production.items()} == expected, 'Production files differ')
    registrations = read(ROOT / 'research/a109_H2_canonical_inputs/training_results.json')
    require(ab_vision.weights_file() == args.weights.resolve(), 'Pinned vision weights unavailable')
    results = {}
    for seed in ('45101', '45102'):
        folder = args.training / seed
        archive, inner = folder / 'train.zip', folder / 'inner_splits.json'
        registered = registrations[seed]
        require(sha(archive) == registered['train_archive_sha256'], 'Training pack changed')
        task = load_development_task(archive)
        spec = TaskSpec.from_task(task).record()
        require(task.X_val is None and task.y_val is None and task.X_train.shape == (2500, 64, 64, 3)
                and len(np.unique(task.y_train)) == 10 and len(task.y_train) == registered['train_rows'],
                'Unexpected training contract')
        capabilities = registered_capabilities(spec, len(task.y_train))
        for arm in ('L0', 'L1'):
            prior = read(ROOT / f'research/a115_H_entry/H2_{seed}_{arm}_manifest.json')
            require(prior['task_contract'] == spec and prior['capabilities'] == capabilities,
                    'Methods or task contract differ from the engineering record')
        require(ab_vision.NAME in capabilities, 'Frozen vision candidate unavailable')
        partition = registered_partitioner(task, inner, registered['inner_manifest_sha256'])
        verified = {}
        for item in read(inner)['splits']:
            previous = registered['splits'][item['id']]
            fit, held, facts = partition(task, item['seed'])
            require(item['seed'] == previous['seed'] and len(fit) == previous['train_rows'] == 2000
                    and len(held) == previous['internal_eval_rows'] == 500
                    and item['row_ids_sha256'] == previous['row_ids_sha256']
                    and item['label_sha256'] == previous['label_sha256']
                    and facts['class_domain_preserved'], 'Registered training partition differs')
            verified[item['id']] = dict(seed=item['seed'], training_rows=len(fit), internal_rows=len(held),
                                        row_ids_sha256=item['row_ids_sha256'], label_sha256=item['label_sha256'])
        require(set(verified) == set(registered['splits']), 'Expected all registered splits')
        if not results:
            import torch

            network, preprocess = ab_vision._network(args.weights.resolve())
            rgb = np.ascontiguousarray(task.X_train[:1].transpose(0, 3, 1, 2))
            with torch.inference_mode():
                features = network(preprocess(torch.from_numpy(rgb)))
            require(tuple(features.shape) == (1, 576) and bool(features.isfinite().all()),
                    'Frozen forward pass failed')
        results[seed] = dict(train_archive_sha256=sha(archive), inner_manifest_sha256=sha(inner),
                             rows=2500, classes=10, candidates=list(capabilities), splits=verified)
    receipt = dict(shared_commit='a36296f550c85e49376caedd0149d21caf94dd63', source_files=len(production),
                   plan_sha256=sha(PLAN), script_sha256=sha(Path(__file__)), inputs=results,
                   vision_weight_sha256=ab_vision.WEIGHT_SHA256, frozen_forward_training_images=1,
                   frozen_forward_dimensions=576, classifier_fits=0, model_api_calls=0,
                   outer_features_read=False, outer_labels_read=False, official_H2_score=None)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('training', 'weights', 'output'):
        parser.add_argument('--' + flag, type=Path, required=True)
    verify(parser.parse_args())
