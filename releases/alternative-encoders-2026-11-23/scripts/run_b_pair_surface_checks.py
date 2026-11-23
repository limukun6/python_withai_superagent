"""Test column symmetry and surface form sensitivity without retraining a predictor."""
import argparse
import csv
import io
import json
from pathlib import Path
import unicodedata
import zipfile

import numpy as np

from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_alternative_encoders import frozen_save
from scripts.run_b_encoder_vote_followup import classes, verified_export
from scripts.run_b_local_component_pair import check, packets, read
from superagent.ab_run import sha

PLAN = Path(__file__).resolve().parents[1] / 'docs/batches/B_pair_surface_checks_1123_zh.md'
EXPECTED = {
    'headlines': (1693, '1d96a64ad65bbaa4b3010174cb1b4e6684faf9a0d5dea4357c8e8069cc98b8ff'),
    'quotes': (450, 'fdebbff1e8a9a707747b1710278713a76efb8bc8a441c9576f1e7ebf83a6aae5'),
}


def plain(text):
    text = unicodedata.normalize('NFKC', text).casefold()
    return ' '.join(''.join(' ' if unicodedata.category(c).startswith('P') else c for c in text).split())


def pairs(file):
    with zipfile.ZipFile(file) as z, z.open('rows.csv') as stream:
        source = csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig'))
        check(source.fieldnames == ['option_a', 'option_b'], 'Expected an unlabelled input')
        return [(r['option_a'], r['option_b']) for r in source]


def run(a):
    a.out.mkdir(parents=True, exist_ok=True)
    hp = read(a.inputs/'protocol.json')
    qp = read(a.quotes/'protocol.json')
    he = read(a.headlines/'exports_frozen.json')
    qe = read(a.quotes/'outer_export.json')
    setups = {
        'headlines': (a.inputs, a.headlines, hp['feature_sha'], hp['locked_scores'], he['model'], he['development']),
        'quotes': (a.quotes, a.quotes, qp['features'], qp['scoring'], qe['model'], qe['predictions']),
    }
    env = offline_environment(a.workspace)
    data, seals = {}, {}
    protocol = dict(plan=sha(PLAN), runner=sha(Path(__file__)), groups_sha=hp['locked_groups'],
                    sealed_inputs={k: list(v[2:]) for k, v in setups.items()}, fits=0, model_api_calls=0)
    frozen_save(a.out/'protocol.json', protocol)
    for task, (input_dir, model_dir, feat, _, model_sha, base) in setups.items():
        inputs = input_dir/'features.zip'
        model = model_dir/'autonomous'
        source = pairs(inputs)
        check(sha(inputs) == feat and sha(model/'selected_model.joblib') == model_sha and
              (len(source), base) == EXPECTED[task], 'Unexpected underlying data or model')
        data[task] = source
        seals[task] = {'original': base}
        variants = {'reverse': [(r, l) for l, r in source], 'plain': [(plain(l), plain(r)) for l,r in source]}
        for variant, rows in variants.items():
            target = a.out/task/variant
            target.mkdir(parents=True, exist_ok=True)
            file = target/'features.zip'
            if not file.exists():
                packets(rows, range(len(rows)), False, file)
            check(pairs(file) == rows, 'Transformed rows differ')
            seals[task][variant] = verified_export(model, target, env, target)
    frozen_save(a.out/'exports.json', seals)
    result = {}
    for task, (input_dir, model_dir, _, score_sha, _, _) in setups.items():
        scoring = input_dir/'scoring_only.json'
        check(sha(scoring) == score_sha, 'Evaluation data changed')
        truth = read(scoring)
        if task == 'headlines':
            check(sha(a.inputs/'groups_only.json') == hp['locked_groups'], 'Headline groups changed')
            groups = read(a.inputs/'groups_only.json')
        else:
            groups = truth['components']
        y = np.array(truth['labels'])
        h = {}
        for name, digest in seals[task].items():
            folder = model_dir if name == 'original' else a.out/task/name
            h[name] = classes(folder/'autonomous_export_0/rows.csv', digest, len(y))
        base, new = h['original'] == y, h['plain'] == y
        diff = np.array([len(l.split())-len(r.split()) for l,r in data[task]])
        strata = {name: dict(rows=int(mask.sum()), correct=int(base[mask].sum()))
                  for name, mask in [('equal', diff == 0), ('unequal', diff != 0)]}
        result[task] = dict(rows=len(y), groups=len(set(groups)), original_correct=int(base.sum()),
            swapped_complements=int((h['original'] == 1-h['reverse']).sum()),
            plain_correct=int(new.sum()), plain_only=int((new & ~base).sum()), original_only=int((base & ~new).sum()),
            normalized_identical=sum(plain(l) == plain(r) for l,r in data[task]),
            plain_minus_original=paired(new.astype(int)-base.astype(int), np.array(groups)), word_count=strata,
            longer_correct_with_half_ties=float(((diff[y == 1]>0).sum() + (diff[y == 0]<0).sum()) + .5*(diff==0).sum()))
    frozen_save(a.out/'results.json', result)
    frozen_save(a.out/'completion.json', dict(protocol=sha(a.out/'protocol.json'), results=sha(a.out/'results.json'),
        exports=sha(a.out/'exports.json'), fits=0, model_api_calls=0, course_designated_used=False))
    print(json.dumps(result))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('workspace','inputs','headlines','quotes','out'):
        p.add_argument('--'+key, type=lambda v: Path(v).resolve(), required=True)
    run(p.parse_args())
