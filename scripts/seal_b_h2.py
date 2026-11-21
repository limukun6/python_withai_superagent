"""Seal source-relative sample IDs for two image transfers. Fits and scores no models."""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
from importlib.metadata import version
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from scripts import run_b1_ab_acceptance as fixture
from scripts.run_b1_05_pair_pilot import identities_file
from superagent.data import TrainingTask
from superagent.splits import grouped_partition


SPATIAL_REVISION = '1ce6f1bfb56db63fd91b6ecc466ea67f2509774c'
SPATIAL = {
    'train': (16200, '2db7d455afb8dcbca898ea19a00f1f90c091734efdbba89e22aaf24056da243f', 250),
    'val': (5400, '6c758477604b7057a0fd990d7f6327b63b99a6725aac11a6a9d0174a7fdd8f0b', 50),
}
SIZE = 94658721
MD5 = 'f46e308c4d50d4bf32fedad2d3d62f3b'
OUTER = (45101, 45102)
INNER = (707, 808, 909)
SALT = 'aiaa3102-h2-eurosat-v1'


def rank(name):
    return (fixture.sha((SALT + '/' + name).encode()), name)


def source_hashes(archive):
    if archive.stat().st_size != SIZE:
        raise ValueError('RGB archive size differs from the author release')
    md5, sha = hashlib.md5(), hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            md5.update(chunk)
            sha.update(chunk)
    if md5.hexdigest() != MD5:
        raise ValueError('RGB archive differs from author checksum')
    return sha.hexdigest()


def spatial_ids(source, region):
    count, digest, _ = SPATIAL[region]
    data = (source / f'eurosat-spatial-{region}.txt').read_bytes()
    names = data.decode('utf-8').splitlines()
    if fixture.sha(data) != digest or len(set(names)) != len(names) or len(names) != count:
        raise ValueError('Unexpected spatial source identity: ' + region)
    return names


def read_region(archive, members, names, quota, seen, pixels):
    classes = defaultdict(list)
    for name in names:
        classes[name.rsplit('_', 1)[0]].append(name)
    if len(classes) != 10:
        raise ValueError('Spatial region lost one of the ten classes')
    sampled = {seed: [] for seed in OUTER}
    duplicate_count = 0
    for label in sorted(classes):
        chosen = []
        for name in sorted(classes[label], key=rank):
            with Image.open(io.BytesIO(archive.read(members[name]))) as im:
                if im.size != (64, 64) or im.mode != 'RGB' or im.format != 'JPEG':
                    raise ValueError('Unexpected source image format: ' + name)
                value = np.array(im)
            key = fixture.sha(value.tobytes())
            if key in seen:
                duplicate_count += 1
                continue
            seen.add(key)
            pixels[name] = value
            chosen.append(name)
            if len(chosen) == quota * len(OUTER):
                break
        if len(chosen) != quota * len(OUTER):
            raise ValueError('Insufficient unique images for ' + label)
        for i, seed in enumerate(OUTER):
            sampled[seed].extend(chosen[i * quota:(i + 1) * quota])
    return {s: sorted(names, key=rank) for s, names in sampled.items()}, duplicate_count


def seal(source, image_zip, output):
    output.mkdir(parents=True, exist_ok=False)
    fixture.write(output / 'protocol.json', dict(dataset='EuroSAT RGB', revision='10.5281/zenodo.7711810',
        source_url='https://zenodo.org/records/7711810', author_license='https://github.com/phelber/EuroSAT#license',
        source_image_count=27000, source_bytes=SIZE, author_md5=MD5,
        spatial_revision=SPATIAL_REVISION, spatial_sources=SPATIAL, sort_salt=SALT, outer=OUTER, inner=INNER,
        input='64 x 64 RGB uint8', primary_metric='accuracy', group='exact decoded pixels',
        plan_sha256=fixture.sha((fixture.ROOT / 'docs/batches/BH_02_seal_plan.md').read_bytes()),
        driver_sha256=fixture.sha(Path(__file__).read_bytes()),
        group_code_sha256=fixture.sha((fixture.ROOT / 'superagent/splits.py').read_bytes()),
        packages={k: version(k) for k in ('numpy', 'Pillow', 'scikit-learn')},
        fits_allowed=0, predictions_allowed=0, scoring=False))
    try:
        sha = source_hashes(image_zip)
        ids = {region: spatial_ids(source, region) for region in SPATIAL}
        if set(ids['train']) & set(ids['val']):
            raise ValueError('Spatial source lists overlap')
        seen, pixels, selected, duplicates = set(), {}, {}, {}
        with zipfile.ZipFile(image_zip) as archive:
            names = [n for n in archive.namelist() if n.lower().endswith('.jpg') and not n.startswith('__MACOSX/')]
            members = {Path(n).name: n for n in names}
            if len(members) != 27000 or len(members) != len(names):
                raise ValueError('Unexpected or ambiguous RGB archive members')
            for region, (_, _, quota) in SPATIAL.items():
                selected[region], duplicates[region] = read_region(archive, members, ids[region], quota, seen, pixels)
        all_ids, summary = {}, {}
        for seed in OUTER:
            train, held = selected['train'][seed], selected['val'][seed]
            labels = [n.rsplit('_', 1)[0] for n in train]
            task = TrainingTask('image', np.stack([pixels[n] for n in train]), np.asarray(labels))
            record = dict(outer_train_source_ids=train, outer_holdout_source_ids=held, inner={})
            partitions = {}
            for inner in INNER:
                fit, search, _ = grouped_partition(task, inner)
                record['inner'][str(inner)] = dict(train_source_ids=[train[i] for i in fit],
                                                   search_source_ids=[train[i] for i in search])
                partitions[str(inner)] = fixture.digest(record['inner'][str(inner)])
            if len(set(partitions.values())) != len(INNER):
                raise ValueError('Inner partitions repeat')
            all_ids[str(seed)] = record
            summary[str(seed)] = dict(train=len(train), held=len(held), train_class_counts=dict(Counter(labels)),
                held_class_counts=dict(Counter(n.rsplit('_',1)[0] for n in held)), inner_row_id_sha256=partitions)
        if len(seen) != sum(len(ids[s]) for ids in selected.values() for s in OUTER):
            raise ValueError('Final source identities have overlapping pixels')
        source_id_sha = identities_file(output / 'source_ids.json.gz', all_ids)
        fixture.write(output / 'results.json', dict(archive_sha256=sha, source_ids_sha256=source_id_sha,
            selected_decoded_pixels_sha256=fixture.digest({n: fixture.sha(p.tobytes()) for n,p in sorted(pixels.items())}),
            skipped_matching_pixels=duplicates, partitions=summary, unique_registered_images=len(seen)))
        fixture.write(output / 'completion.json', dict(status='sealed', outer_partitions=2, fits=0, predictions=0,
            performance_scores=0, api_requests=0, spatial_test_list_opened=False, images_outside_train_val_decoded=0))
    except Exception as error:
        fixture.write(output / 'completion.json', dict(status='failed', reason=str(error), fits=0, performance_scores=0))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--rgb-zip', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    seal(args.source, args.rgb_zip, args.output)
