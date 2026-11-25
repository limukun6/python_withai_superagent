"""Build anonymous packets for the preregistered planning-policy experiment."""
import csv
import gzip
import io
import struct
import tarfile
import zipfile

import numpy as np
from sklearn.datasets import load_wine, make_circles, make_classification, make_hastie_10_2
from sklearn.model_selection import train_test_split

from scripts.audit_b_kernel_external_movies import ARCHIVE, MEMBERS
from scripts.run_planner_self_review import distinct, packet
from scripts.unified_released_study import require, sha


SPLIT_SEED = 655406


def sample(ids, labels, size):
    return train_test_split(ids, train_size=size, stratify=labels[ids], random_state=SPLIT_SEED)[0]


def normalized(text):
    return ' '.join(text.casefold().split())


def cases(downloads, archive, prior):
    wine = load_wine()
    yield 'u11', 'tabular', wine.data, wine.target
    rng = np.random.default_rng(655401)
    x, y = make_circles(800, noise=.17, factor=.47, random_state=655401)
    rotation, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    x = (x @ rotation) * [1.6, .8] + [1.1, -.7]
    yield 'u12', 'tabular', np.column_stack((x, rng.normal(size=(800, 6)))), y

    rng = np.random.default_rng(655402)
    x = rng.integers(0, 8, size=(1000, 9))
    y = np.where(x[:, 0] < 4, x[:, 1] >= x[:, 2], x[:, 3] // 2 == x[:, 4] // 2)
    y = np.logical_xor(y, rng.random(len(y)) < .08).astype(int)
    yield 'u21', 'categorical', np.char.add('v', x.astype(str)), y

    previous = set()
    for filename in ('p.zip', 'q.zip'):
        with zipfile.ZipFile(prior/'paired/n29'/filename) as z, z.open('rows.csv') as f:
            previous.update(normalized(r['document']) for r in csv.DictReader(io.TextIOWrapper(f, encoding='utf-8')))
    with zipfile.ZipFile(downloads/'sms.zip') as z:
        rows = [row.split('\t', 1) for row in z.read('SMSSpamCollection').decode('utf-8-sig').splitlines() if row]
    x, y = np.asarray([r[1] for r in rows]), np.asarray([r[0] for r in rows])
    ids = [i for i in distinct(x, y) if normalized(x[i]) not in previous]
    selected = sample(np.array(ids), y, 800)
    require(not (previous & {normalized(t) for t in x[selected]}), 'Historical SMS text overlap')
    yield 'u22', 'text', x[selected], y[selected]

    rng = np.random.default_rng(655403)
    y = rng.integers(0, 3, 800)
    offset = rng.integers(3, 9, (800, 2, 1, 1))
    row, col = np.indices((12, 12))
    dy, dx = row - offset[:, 0], col - offset[:, 1]
    patterns = np.select([y[:, None, None] == 0, y[:, None, None] == 1],
                         [np.abs(dy) < 1.5, (np.abs(dy) < 1.5) | (np.abs(dx) < 1.5)],
                         default=np.abs(dx - dy) < 1.5)
    intensity = rng.uniform(27, 86, size=(800, 1, 1))
    images = np.clip(rng.normal(100, 45, size=patterns.shape) + intensity * patterns, 0, 255)
    yield 'u31', 'image', images.astype('uint8'), y
    x, y = make_classification(n_samples=900, n_features=32, n_informative=5, n_redundant=2, shuffle=False,
                               weights=[.73, .27], flip_y=.06, class_sep=.9, random_state=655404)
    interaction = (x[:, 0] > .25) & (x[:, 3] < -.4)
    yield 'u32', 'tabular', x, np.logical_xor(y, interaction).astype(int)

    with gzip.open(downloads/'fashion-images.gz', 'rb') as f:
        magic, n, h, w = struct.unpack('>IIII', f.read(16))
        require((magic, n, h, w) == (2051, 60000, 28, 28), 'Expected original public training images')
        x = np.frombuffer(f.read(), dtype=np.uint8).reshape(n, h, w)
    with gzip.open(downloads/'fashion-labels.gz', 'rb') as f:
        require(struct.unpack('>II', f.read(8)) == (2049, n), 'Mismatched public training labels')
        y = np.frombuffer(f.read(), dtype=np.uint8)
    subset = np.flatnonzero(np.isin(y, [0, 2, 4, 6]))
    ids = subset[distinct(x[subset], y[subset])]
    selected = sample(ids, y, 1000)
    yield 'v11', 'image', x[selected], y[selected]
    require(sha(archive) == ARCHIVE, 'Cornell public archive digest differs')
    texts, labels = [], []
    with tarfile.open(archive, 'r:gz') as data:
        for name, (label, _) in MEMBERS.items():
            lines = data.extractfile(name).read().decode('latin-1').splitlines()
            texts.extend(lines)
            labels.extend([label] * len(lines))
    x, y = np.asarray(texts), np.asarray(labels)
    selected = sample(distinct(x, y), y, 800)
    yield 'v12', 'text', x[selected], y[selected]
    x, y = make_hastie_10_2(n_samples=1000, random_state=655405)
    yield 'v13', 'tabular', x, (y > 0).astype(int)


def prepare_packets(root, plan, downloads, archive, prior):
    contracts = {c['id']: c['metric'] for cohort in [*plan['waves'], plan['final_panel']] for c in cohort}
    truth_dir = root/'evaluation'
    truth_dir.mkdir()
    output = {}
    for key, modality, x, y in cases(downloads, archive, prior):
        folder = root/key
        folder.mkdir()
        ids = distinct(x, y)
        fit, held = train_test_split(ids, test_size=.25, stratify=y[ids], random_state=SPLIT_SEED)
        require(set(fit).isdisjoint(held), 'Duplicate outer rows')
        metric = contracts[key]
        packet(folder/'p.zip', modality, metric, x[fit], y[fit])
        packet(folder/'q.zip', modality, metric, x[held])
        truth = truth_dir/(key+'.npy')
        np.save(truth, y[held], allow_pickle=False)
        output[key] = dict(metric=metric, train_rows=len(fit), held_rows=len(held),
                           train=sha(folder/'p.zip'), features=sha(folder/'q.zip'), truth=sha(truth))
    require(output.keys() == contracts.keys(), 'Case registration disagrees with packets')
    return output
