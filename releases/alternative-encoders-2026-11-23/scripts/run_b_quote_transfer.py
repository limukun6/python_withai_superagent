"""Score two frozen headline predictors on the original Cornell Movie-Quotes pairs."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import unicodedata
import zipfile

import numpy as np

from scripts.b_frozen_head_frontier import paired
from scripts.probe_b_alternative_encoders import offline_environment
from scripts.run_b_encoder_vote_followup import classes, verified_export
from scripts.run_b_local_component_pair import check, packets, read, save
from superagent.ab_run import sha


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'docs/batches/B_frozen_quote_transfer_1123_zh.md'
SOURCE = 'https://www.cs.cornell.edu/~cristian/memorability_files/cornell_movie_quotes_corpus.zip'
PAIRS = 'moviequotes.memorable_nonmemorable_pairs.txt'
MEMBERS = {
    PAIRS: '6604ba9716a4d669b862d4d55d829f7f2ed0d3d530360789da64f05de8c7284a',
    'README.v1.0.txt': '02864d429c50bc919cf02e05523003dbfec3c273f607c243ba1b38d20dde7dad',
}
INPUT_TRAIN = 'f4e30ae46244196b927b2ea278c2d900ba2c53bcffa7b93f165e47aed1190bd2'
MODELS = {
    'control': 'e6bbed2afa1e541cfe98e9266b43ffcb348f1f137106223d24a373cb593119df',
    'joint': '1ced098c9f8b993f8d02200fd332ee43eda4a7b9613dd63988b0357c5642eccb',
}
CONTRACT = '8536e83eb1cf92d80c8ae6cfb6e2e8274c698f0fabaf94dd298c03572e454b22'


def normalized(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def movie_pairs(archive):
    with zipfile.ZipFile(archive) as z:
        parts = {name: z.read(name) for name in MEMBERS}
    for name, content in parts.items():
        check(hashlib.sha256(content).hexdigest() == MEMBERS[name], 'Cornell member changed: ' + name)
    text = parts[PAIRS].decode('cp1252').replace('\r\n', '\n')
    blocks = re.split(r'\n[ \t]*\n', text.strip())
    check(len(blocks) == 2197, 'Unexpected number of authored records')
    rows, films = [], []
    omitted = {'blank': 0, 'identical': 0}
    for block in blocks:
        film, _, winner, other = block.splitlines()
        sides = [line.partition(' ') for line in (winner, other)]
        check(all(prefix.isdecimal() and space for prefix, space, _ in sides), 'Missing script line identifier')
        winner, other = [value.strip() for _, _, value in sides]
        film = normalized(film)
        check(bool(film), 'Missing film')
        keys = list(map(normalized, (winner, other)))
        if not all(keys):
            omitted['blank'] += 1
            continue
        if keys[0] == keys[1]:
            omitted['identical'] += 1
            continue
        canon = sorted(zip(keys, (winner, other)))
        key = json.dumps(['quote-transfer-1123-v1', film, canon[0][0], canon[1][0]],
                         ensure_ascii=False, separators=(',', ':')).encode('utf-8')
        left = hashlib.sha256(key).digest()[0] % 2
        rows.append((canon[left][1], canon[1-left][1], int(canon[left][0] == keys[0])))
        films.append(film)
    return rows, films, omitted


def title_overlap(training, rows):
    check(sha(training) == INPUT_TRAIN, 'Expected the original headline training packet')
    with zipfile.ZipFile(training) as z, z.open('rows.csv') as f:
        known = {normalized(r[k]) for r in csv.DictReader(io.TextIOWrapper(f, encoding='utf-8-sig'))
                 for k in ('option_a', 'option_b')}
    return sum(any(normalized(t) in known for t in r[:2]) for r in rows)


def prepare(args):
    args.out.mkdir(parents=True, exist_ok=False)
    rows, films, omitted = movie_pairs(args.archive)
    packets(rows, range(len(rows)), False, args.out / 'features.zip')
    save(args.out / 'scoring_only.json', {'winner_left': [r[2] for r in rows], 'films': films})
    for name in MODELS:
        folder = getattr(args, name)
        receipt = read(folder / 'final_selection.json')
        check(sha(folder / 'selected_model.joblib') == receipt['model_sha256'] == MODELS[name] and
              sha(folder / 'task_spec.json') == receipt['task_spec_sha256'] == CONTRACT and
              receipt['training_rows'] == 6608, 'Unexpected frozen ' + name)
    save(args.out / 'protocol.json', dict(
        source=SOURCE, members=MEMBERS, raw_pairs=2197, rows=len(rows), movies=len(set(films)),
        omitted=omitted, headline_train=INPUT_TRAIN, headline_overlap_pairs=title_overlap(args.training, rows),
        models=MODELS, contract=CONTRACT, plan=sha(PLAN), runner=sha(Path(__file__)),
        features=sha(args.out / 'features.zip'), evaluation_record=sha(args.out / 'scoring_only.json'),
        fits=0, model_api_calls=0, movie_bootstrap_draws=5000, formal_test_accessed=False))


def run(args):
    if not args.out.exists():
        prepare(args)
    plan = read(args.out / 'protocol.json')
    check(plan['plan'] == sha(PLAN) and plan['runner'] == sha(Path(__file__)) and
          plan['features'] == sha(args.out / 'features.zip'), 'Frozen protocol changed')
    env = offline_environment(args.workspace)
    files, seals = {}, {}
    for name in MODELS:
        folder = getattr(args, name)
        check(sha(folder / 'selected_model.joblib') == MODELS[name] and
              sha(folder / 'task_spec.json') == CONTRACT, 'Frozen predictor changed')
        seals[name] = verified_export(folder, args.out, env, args.out)
        files[name] = args.out / (folder.name + '_export_0') / 'rows.csv'
    sealfile = args.out / 'exports.json'
    if not sealfile.exists():
        save(sealfile, seals)
    check(read(sealfile) == seals and sha(args.out / 'scoring_only.json') == plan['evaluation_record'],
          'Sealed outputs or evaluation rows changed')
    truth = read(args.out / 'scoring_only.json')
    y, groups = np.array(truth['winner_left']), np.array(truth['films'])
    hits = {name: classes(file, seals[name], plan['rows']) == y for name, file in files.items()}
    summary = {name: dict(correct=int(h.sum()), rows=len(h), accuracy=float(h.mean()),
                          versus_chance=paired(h.astype(float) - .5, groups)) for name, h in hits.items()}
    base, new = hits['control'], hits['joint']
    delta = new.astype(int) - base.astype(int)
    result = dict(scores=summary, movies=plan['movies'], fixed_left_correct=int(y.sum()),
                  fixed_right_correct=int(len(y)-y.sum()), joint_only=int((delta > 0).sum()),
                  control_only=int((delta < 0).sum()), both_correct=int((new & base).sum()),
                  both_wrong=int((~new & ~base).sum()), joint_minus_control=paired(delta, groups),
                  predictions=seals, independent_exports_per_model=2, fits=0, model_api_calls=0)
    target = args.out / 'results.json'
    if not target.exists():
        save(target, result)
        save(args.out / 'completion.json', dict(protocol=sha(args.out / 'protocol.json'),
            results=sha(target), exports=sha(sealfile), fits=0, model_api_calls=0))
    check(read(target) == result and read(args.out / 'completion.json')['results'] == sha(target),
          'Completed result differs')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('workspace', 'archive', 'training', 'control', 'joint', 'out'):
        parser.add_argument('--' + key, type=Path, required=True)
    run(parser.parse_args())
