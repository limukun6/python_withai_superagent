"""Check the actual training reader in a specified checkout using equivalent, anonymous preference tasks."""
import argparse
import csv
from hashlib import sha256
import inspect
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile


def write_task(path, fields, target, description, manifest):
    out = io.StringIO(newline='')
    writer = csv.writer(out)
    writer.writerow([*fields, target])
    for i in range(32):
        writer.writerow([f'The first alternative describes a longer route around test location {i}',
                         f'The second alternative describes the shorter route through district {i}', i % 2])
    with zipfile.ZipFile(path, 'x') as archive:
        archive.writestr('train.csv', out.getvalue())
        archive.writestr('README.md', description + ' Report accuracy.')
        if manifest:
            archive.writestr('manifest.json', json.dumps(dict(schema_version=1, objective='classification',
                metric='accuracy', modality='pair_text', columns=[dict(name=n, type='text') for n in fields],
                target=target, pair_semantics='preference', pair_label_one='left',
                splits={'train': dict(format='csv', path='train.csv')})))


def verify(source):
    sys.path.insert(0, str(source))
    from superagent import data
    if Path(data.__file__).resolve() != source / 'superagent/data.py':
        raise ValueError('Unexpected implementation of the training reader')
    options = {'read_declarations': True} if 'read_declarations' in inspect.signature(data.load_training_task).parameters else {}
    original = ('headline_A', 'headline_B')
    aliases = ('left_text', 'right_text')
    wording = 'Target label_pairwise is 1 if A had the higher rate and 0 if B did.'
    other = ('Each row contains two alternatives, in left then right order. Label 1 means that the left '
             'alternative was preferred, and label 0 means that the right one was preferred.')
    cases = [('original', 'untitled.zip', original, 'label_pairwise', wording, False),
             ('filename_only', 'Task1_coupon.zip', original, 'label_pairwise', wording, False),
             ('prose_without_target_name', 't.zip', original, 'label_pairwise', other, False),
             ('reworded_with_declared_target', 'z.zip', original, 'label_pairwise', 'The target column is label_pairwise. ' + other, False),
             ('renamed_and_reworded', 'y.zip', aliases, 'label', other, False),
             ('generic_manifest', 'x.zip', aliases, 'label', other, True)]
    observed = {}
    with tempfile.TemporaryDirectory() as tmp:
        for case, name, fields, target, prose, manifest in cases:
            path = Path(tmp) / name
            write_task(path, fields, target, prose, manifest)
            try:
                t = data.load_training_task(path, **options)
            except ValueError as first:
                if not (source / 'superagent/format_intake.py').is_file() or not options:
                    observed[case] = dict(kind=None, reason=str(first))
                    continue
                from superagent.format_intake import DECODE_ERRORS, NativeContract, adapt_training
                try:
                    t = adapt_training(path, client=None)
                except NativeContract:
                    observed[case] = dict(kind=None, reason=str(first))
                    continue
                except DECODE_ERRORS as final:
                    observed[case] = dict(kind=None, reason=str(final))
                    continue
            schema = t.spec.get('input_schema', {})
            row = dict(kind=t.kind, semantics=schema.get('pair_semantics'), direction=schema.get('pair_label_one'))
            if t.kind == 'pair_text':
                pairs = [[list(pair), int(y)] for pair, y in zip(t.X_train, t.y_train)]
                row['ordered_training_sha256'] = sha256(json.dumps(pairs).encode()).hexdigest()
            observed[case] = row
    return observed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=lambda s: Path(s).resolve())
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = verify(args.source)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
