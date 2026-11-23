"""Check representation diversity on hashed training titles before deciding which paths to run."""
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_b_local_component_pair import check, read, save
from scripts.run_b_shared_preference_pair import environment
from superagent import ab_run
from superagent.ab_methods import PREFERENCE_CATALOG_ENV, TRANSFER_METHODS
from superagent.ab_sentence import ALTERNATIVES, AlternativeSentenceFeatures, encode, encode_alternative, verified_files
from superagent.external_assets import model_resources, verify_resources


def offline_environment(workspace):
    env = environment(workspace)
    for spec in ALTERNATIVES.values():
        env[spec['env']] = str(workspace / 'work/pretrained_weights' / spec['model'].split('/')[-1])
    return env


def author_vectors(texts, variant):
    import torch
    from transformers import AutoModel, AutoTokenizer
    spec = ALTERNATIVES[variant]
    root = os.environ[spec['env']]
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    model = AutoModel.from_pretrained(root, local_files_only=True, add_pooling_layer=False).eval()
    content = ['query: ' + s for s in texts] if variant == 'e5' else texts
    batch = tokenizer(content, max_length=512, truncation=True, padding=True, return_tensors='pt')
    with torch.inference_mode():
        hidden = model(**batch).last_hidden_state
        if variant == 'e5':
            mask = batch['attention_mask'].bool()
            pooled = hidden.masked_fill(~mask[..., None], 0.).sum(dim=1) / mask.sum(dim=1)[..., None]
        else:
            pooled = hidden[:, 0]
        return torch.nn.functional.normalize(pooled, dim=1).numpy()


def run(inputs, control, output):
    output.mkdir(parents=True, exist_ok=False)
    prior = read(control / 'protocol.json')
    check(prior['interfaces']['files']['anonymous.zip'] == ab_run.sha(inputs / 'anonymous.zip'), 'Training packet changed')
    with zipfile.ZipFile(inputs / 'anonymous.zip') as archive, archive.open('rows.csv') as rows:
        titles = {row[key] for row in csv.DictReader(io.TextIOWrapper(rows, encoding='utf-8-sig'))
                  for key in ('option_a', 'option_b')}
    sample = sorted(titles, key=lambda title: (hashlib.sha256(title.encode()).digest(), title))[:512]
    check(len(sample) == 512, 'Too few distinct training titles')
    options = {}
    neighbors = {}
    task = read(control / 'deep/run_manifest.json')['task_contract']
    with patch.dict(os.environ, {PREFERENCE_CATALOG_ENV: 'deep'}):
        standard = ab_run.registered_capabilities(task, prior['training_rows'], 'full')
    check(len(standard) == prior['sizes']['deep'] == 33, 'Control menu changed')
    for name in ('deep', 'e5', 'bge'):
        if name == 'deep':
            vectors, _ = encode(sample, deep=True)
        else:
            variant, candidate = TRANSFER_METHODS['deep-' + name]
            spec = ALTERNATIVES[variant]
            features = AlternativeSentenceFeatures('pair_text', variant)
            resources, _ = model_resources(features)
            check([r['model'] for r in resources] == [spec['model']], 'Reload resource identity')
            verify_resources(resources)
            with patch.dict(os.environ, {PREFERENCE_CATALOG_ENV: 'deep-' + name}):
                expanded = ab_run.registered_capabilities(task, prior['training_rows'], 'full')
                check(candidate in expanded and [(k,v) for k,v in expanded.items() if k != candidate] == list(standard.items()),
                      'Common menu configs or order differ')
                with patch.dict(os.environ, {spec['env']: ''}):
                    missing = ab_run.registered_capabilities(task, prior['training_rows'], 'full')
                    check(missing == standard and verified_files(spec['env'],spec['files']) is None, 'Missing encoder exposed')
            vectors, _ = encode_alternative(sample, variant)
            reference = author_vectors(sample[:12], variant)
            deviation = float(np.abs(reference - vectors[:12]).max())
            np.testing.assert_allclose(vectors[:12], reference, atol=3e-6, rtol=2e-5)
            normal = features.embedding_features(vectors[:12])[0].toarray()
            swapped = features.embedding_features(vectors[:12].reshape(6,2,-1)[:,::-1].reshape(12,-1))[0].toarray()
            np.testing.assert_array_equal(normal, -swapped)
            options[name] = dict(candidate=candidate, reference_max_abs=deviation, menu_size=len(expanded), resources=resources)
        check(vectors.shape == (512,384) and np.isfinite(vectors).all(), 'Wrong vectors')
        np.testing.assert_allclose(np.linalg.norm(vectors,axis=1),1.,atol=5e-6)
        sims = vectors @ vectors.T
        np.fill_diagonal(sims, -np.inf)
        neighbors[name] = np.argsort(-sims, axis=1, kind='stable')[:,:10]
    overlap = {}
    for a,b in [('e5','deep'),('bge','e5'),('bge','deep')]:
        overlap[a+'_'+b] = sum(len(set(x).intersection(y)) for x,y in zip(neighbors[a],neighbors[b])) / (512*10)
    arms = ['deep-e5']
    if max(overlap['bge_e5'], overlap['bge_deep']) < .80:
        arms.append('deep-bge')
    result = dict(training_packet_sha256=ab_run.sha(inputs/'anonymous.zip'), sample_sha256=hashlib.sha256(
        json.dumps(sample,ensure_ascii=False,separators=(',',':')).encode()).hexdigest(), titles=512,
        distinct_training_titles=len(titles), mean_top10_overlap=overlap, technical=options,
        order=arms, supervised_fits=0, model_api=0, target_values_used=False,
        method_protocol_sha256=ab_run.sha(ROOT/'docs/batches/B_alternative_encoders_protocol_1123_zh.md'))
    save(output/'probe.json', result)
    print(json.dumps(dict(overlap=overlap, order=arms)),flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ('workspace','inputs','control','output'):
        p.add_argument('--'+arg,type=Path,required=True)
    args=p.parse_args()
    with patch.dict(os.environ,offline_environment(args.workspace.resolve())):
        run(args.inputs.resolve(),args.control.resolve(),args.output.resolve())
