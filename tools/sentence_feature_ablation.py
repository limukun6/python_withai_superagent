"""Isolate the added sentence features in a frozen hybrid text classifier."""
import argparse
import hashlib
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.base import clone

from superagent.data import load_training_task
from superagent.method_choice import TEXT_MINILM
from superagent.minilm_features import feature_usage
from superagent.pretrained_choice import digest
from superagent.splits import take
from tools.frontier_study import expect, receipt
from tools.pretrained_study import read, train_only, write
from tools.soft_vote_diagnostic import freeze_file


def random_sentences(texts):
    cache = {}
    for text in dict.fromkeys(texts):
        seed = int.from_bytes(hashlib.sha256(('feature-control-v1|' + text).encode()).digest()[:16], 'big')
        vector = np.random.default_rng(seed).standard_normal(384).astype('float32')
        cache[text] = vector / np.linalg.norm(vector)
    return np.stack([cache[text] for text in texts])


def representations(model, rows, kind):
    original = model.features(rows).tocsr()
    lexical, dense = original[:, :-384], original[:, -384:]
    expect(lexical.shape[1] > 0 and dense.shape[1] == 384, 'Unexpected hybrid feature layout')
    if kind == 'pair_text':
        random = random_sentences([s for pair in rows for s in pair])
        random = random[::2] - random[1::2]
    else:
        random = random_sentences(list(rows))
    return dict(hybrid=original, lexical_only=lexical, sentence_only=dense,
                lexical_with_random=hstack([lexical, csr_matrix(random)],format='csr'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('reference','archive','output'):
        parser.add_argument('--' + flag, required=True, type=Path)
    parser.add_argument('--arm', default='text-encoder')
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    os.environ['SUPERAGENT_OFFLINE'] = '1'
    os.environ.pop('SUPERAGENT_PRETRAINED_RUNTIME', None)
    root = args.output.resolve()
    root.mkdir(exist_ok=True, parents=True)
    protocol = read(args.reference / 'protocol.json')
    expect(protocol['archive_sha256'] == digest(args.archive), 'Archive changed')
    sources = [Path(block[args.arm]) for block in protocol['locations']]
    records = []
    for source in sources:
        trace = receipt(source)
        expect(trace['winner'] == TEXT_MINILM, 'This comparison requires the same winning hybrid in every fold')
        records.append(dict(final_sha256=trace['model_sha256'],fit_payload_sha256=digest(source/'refit_data.joblib')))
    freeze_file(root / 'protocol.json',dict(reference_sha256=digest(args.reference/'protocol.json'),arm=args.arm,
        records=records,source_sha256=digest(Path(__file__)),
        rule='Keep the original fold-fitted TF-IDF vocabulary, style transform, training examples, augmentation, solver '
             'and C. Refit the same linear classifier using lexical plus true vectors, lexical alone, vectors alone, '
             'or lexical plus deterministic unit random vectors keyed by text, 384 dimensions and the original pair orientation. '
             'Freeze all fits before scoring. Require the rebuilt true hybrid to reproduce the deployed predictions.'))
    if args.freeze:
        return
    opened = []
    with train_only(args.archive, opened):
        task = load_training_task(args.archive)
    for i, source in enumerate(sources):
        dest = root / f'fold_{i}'
        dest.mkdir(exist_ok=True)
        model = joblib.load(source/'selected_model.joblib')['model']
        ids = protocol['indices'][i]['train']
        payload = joblib.load(source/'refit_data.joblib')
        x = take(task.X_train,ids)
        y = task.y_train[ids]
        np.testing.assert_array_equal(payload['X_train'], x)
        np.testing.assert_array_equal(payload['y_train'], y)
        before = feature_usage(source)
        blocks = representations(model, x, task.kind)
        expect(feature_usage(source) == before, 'Unexpected training-side encoder work')
        train_labels = np.concatenate([y,1-y]) if task.kind=='pair_text' else y
        for name, block in blocks.items():
            path = dest / f'{name}.joblib'
            proof = path.with_suffix('.json')
            if proof.exists():
                expect(digest(path) == read(proof)['sha256'], 'Previously fitted classifier changed')
                continue
            matrix = vstack([block,-block]) if task.kind=='pair_text' else block
            fitted = clone(model.classifier).fit(matrix,train_labels)
            joblib.dump(fitted,path)
            freeze_file(proof,dict(sha256=digest(path),features=matrix.shape[1],classifier_fits=1))
    folds = []
    for i, source in enumerate(sources):
        dest = root / f'fold_{i}'
        ids = protocol['indices'][i]['outer']
        y = task.y_train[ids]
        model = joblib.load(source/'selected_model.joblib')['model']
        cache = source.parent / (args.arm + '_prediction_features')
        model.bind_prediction_weights(read(cache/'pretrained_runtime.json')[model.weight_setting],cache)
        before = feature_usage(cache)
        blocks = representations(model,take(task.X_train,ids),task.kind)
        expect(before == feature_usage(cache),'Unexpected outer encoder work')
        pred = {}
        for name, block in blocks.items():
            path = dest / f'{name}.joblib'
            expect(digest(path)==read(path.with_suffix('.json'))['sha256'],'Frozen classifier changed')
            pred[name] = joblib.load(path).predict(block)
        recorded = read(args.reference/f"fold_{protocol['fold_numbers'][i]-1}"/'outer_predictions.json')
        expect(recorded['rows']==ids,'Reference row order changed')
        np.testing.assert_array_equal(y,recorded['truth'])
        np.testing.assert_array_equal(pred['hybrid'],recorded[args.arm])
        write(dest/'outer_predictions.json',dict(rows=ids,truth=y.tolist(),**{k:p.tolist() for k,p in pred.items()}))
        folds.append(dict(rows=len(ids),correct={k:int((p==y).sum()) for k,p in pred.items()}))
    total = {k:sum(f['correct'][k] for f in folds) for k in folds[0]['correct']}
    write(root/'results.json',dict(folds=folds,rows=sum(f['rows'] for f in folds),correct=total,
        additional_diagnostic_fits=4*len(folds),additional_encoder_forwards=0,new_api_calls=0,training_members=opened))
    print(total)


if __name__=='__main__':
    main()
