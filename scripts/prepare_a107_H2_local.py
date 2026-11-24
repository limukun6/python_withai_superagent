"""One bounded public download and audited training-only H2 preparation; no fits or API."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from unittest.mock import patch
import zipfile

import requests

from scripts.a_batch_common import hashes, save
from scripts.prepare_a105_H2_training_inputs import URL, SIZE, EXPECTED, ROOT
from scripts.run_a10_real_task_pilot import read, sha


def response_mode(status, content_range, offset):
    if status == 200:
        return 'full'
    if status == 206 and content_range == f'bytes {offset}-{SIZE - 1}/{SIZE}':
        return 'resume'
    raise ValueError(f'Unexpected HTTP status/range: {status}, {content_range!r}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('prefix', 'assets', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    before = hashes(ROOT)
    prefix_hash = sha(args.prefix)
    offset = args.prefix.stat().st_size
    references = ['scripts/prepare_b_h2_training_inputs.py', 'scripts/seal_b_h2.py',
                  'research/bh02_image_seal/source_ids.json.gz',
                  'research/bh_03_training_inputs/results.json']
    save(args.output / 'protocol.json', dict(batch='A107', source_files=before,
         runner_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT / 'docs/batches/A107_H2_local_recovery_plan.md'),
         reference_sha256={p: sha(ROOT / p) for p in references},
         source_url=URL, expected_bytes=SIZE, expected_sha256=EXPECTED,
         prefix_bytes=offset, prefix_sha256=prefix_hash, download_window_seconds=1800,
         request_timeout_seconds=60, automatic_retry=False,
         actual_classifier_fits=0, real_API_requests=0, holdout_image_access_allowed=False))
    accesses = []
    try:
        if not 0 < offset < SIZE:
            raise ValueError('Invalid old prefix size')
        partial = args.output / 'EuroSAT_RGB.zip.partial'
        download_start = time.monotonic()
        last_progress = download_start
        with requests.Session() as session:
            session.proxies.update(http='http://127.0.0.1:7897', https='http://127.0.0.1:7897')
            with session.get(URL, headers={'Range': f'bytes={offset}-', 'Accept-Encoding': 'identity'},
                             stream=True, timeout=60) as response:
                response.raise_for_status()
                mode = response_mode(response.status_code, response.headers.get('Content-Range'), offset)
                if response.headers.get('Content-Encoding', 'identity') != 'identity':
                    raise ValueError('Encoded response cannot preserve byte ranges')
                save(args.output / 'download_response.json', dict(status=response.status_code, mode=mode,
                     content_range=response.headers.get('Content-Range'),
                     content_length=response.headers.get('Content-Length')))
                count = offset if mode == 'resume' else 0
                if mode == 'resume':
                    shutil.copyfile(args.prefix, partial)
                print(json.dumps(dict(mode=mode, initial_bytes=count)), flush=True)
                with partial.open('ab' if mode == 'resume' else 'xb') as stream:
                    for chunk in response.iter_content(256 * 1024):
                        elapsed = time.monotonic() - download_start
                        if elapsed > 1800:
                            raise TimeoutError('Public download window elapsed; no retry')
                        if count + len(chunk) > SIZE:
                            raise ValueError('Archive exceeds pinned size')
                        stream.write(chunk)
                        count += len(chunk)
                        if time.monotonic() - last_progress >= 30:
                            progress = dict(bytes=count, expected_bytes=SIZE, elapsed_seconds=elapsed)
                            save(args.output / 'download_progress.json', progress)
                            print(json.dumps(progress), flush=True)
                            last_progress = time.monotonic()
        if partial.stat().st_size != SIZE or sha(partial) != EXPECTED:
            raise ValueError('Whole archive does not match registered bytes/SHA256')
        from scripts.seal_b_h2 import source_hashes
        if source_hashes(partial) != EXPECTED:
            raise ValueError('Author checksum mismatch')
        archive_path = args.output / 'EuroSAT_RGB.zip'
        partial.rename(archive_path)
        print('Verified full archive SHA256 and author MD5; starting training-only preparation', flush=True)
        download_seconds = time.monotonic() - download_start
        records = json.loads(gzip.decompress((ROOT / references[2]).read_bytes()))
        allowed = {n for r in records.values() for n in r['outer_train_source_ids']}
        forbidden = {n for r in records.values() for n in r['outer_holdout_source_ids']}
        if len(allowed) != 5000 or allowed & forbidden:
            raise ValueError('Invalid sealed training membership')
        original_open = zipfile.ZipFile.open

        def guarded_open(archive, name, mode='r', *pos, **kw):
            if Path(archive.filename).resolve() == archive_path.resolve():
                member = name.filename if isinstance(name, zipfile.ZipInfo) else name
                short = Path(member).name
                if mode != 'r' or short not in allowed:
                    raise ValueError('Attempt to open non-training original archive member')
                accesses.append(short)
            return original_open(archive, name, mode, *pos, **kw)

        from scripts.prepare_b_h2_training_inputs import prepare
        with patch.object(zipfile.ZipFile, 'open', guarded_open):
            prepare(archive_path, args.output / 'training')
        actual = read(args.output / 'training/results.json')
        expected = read(ROOT / 'research/bh_03_training_inputs/results.json')
        if set(actual) != set(expected):
            raise ValueError('Outer registrations differ')
        for seed, row in actual.items():
            for key in ('train_archive_sha256', 'inner_manifest_sha256'):
                if row[key] != expected[seed][key]:
                    raise ValueError('B registered input identity differs: ' + seed + '/' + key)
        if len(accesses) != 5000 or set(accesses) != allowed:
            raise ValueError('Original archive training access audit differs')
        os.environ['SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS'] = str(args.assets / 'mobilenet_v3_small-047dcff4.pth')
        from superagent.ab_run import registered_capabilities
        from superagent.contracts import TaskSpec
        from superagent.data import load_training_task
        menus = {}
        for seed in actual:
            task = load_training_task(args.output / 'training' / seed / 'train.zip')
            spec = TaskSpec.from_task(task).record()
            caps = registered_capabilities(spec, len(task.y_train), 'full')
            if len(caps) != 9 or 'image_mobilenet_v3_small_linear' not in caps:
                raise ValueError('Expected full9 image capability menu unavailable')
            menus[seed] = dict(task_spec=spec, capabilities=caps, rows=len(task.y_train))
        unchanged = hashes(ROOT) == before and sha(args.prefix) == prefix_hash
        if not unchanged:
            raise ValueError('Production source or preserved A105 prefix changed')
        save(args.output / 'results.json', dict(registered_inputs=actual, menus=menus,
             original_source_sha256=EXPECTED, download_seconds=download_seconds,
             original_member_reads=len(accesses), unique_training_members=len(set(accesses)),
             opened_member_ids_sha256=hashlib.sha256(json.dumps(sorted(accesses)).encode()).hexdigest(),
             outer_holdout_image_members_opened=0, source_unchanged=True, old_prefix_unchanged=True))
        save(args.output / 'completion.json', dict(status='prepared', training_photos_verified=5000,
             inner_partitions_verified=6, training_archives_match_B=True, inner_manifests_match_B=True,
             actual_classifier_fits=0, real_API_requests=0, model_predictions=0,
             holdout_image_members_opened=0, formal_H_strategy_frozen=False,
             elapsed_seconds=time.monotonic() - started))
        print('H2 local inputs prepared: 5000 training images, six partitions, full9 menus; zero fits/API', flush=True)
    except Exception as error:
        save(args.output / 'completion.json', dict(status='stopped', error=str(error),
             automatic_retry=False, actual_classifier_fits=0, real_API_requests=0,
             original_member_reads=len(accesses), elapsed_seconds=time.monotonic() - started))
        raise


if __name__ == '__main__':
    main()
