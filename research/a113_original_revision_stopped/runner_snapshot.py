"""Acquire only B-pinned public L12 bytes, without model API calls or automatic retry."""
import argparse
from pathlib import Path
import shutil
import time
import requests
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import sha

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'sentence-transformers/all-MiniLM-L12-v2'
REVISION = '52f9d2dd711ffc3434e6cff17284b13a11629a90'
FILES = {
    'config.json': (615, 'bc451f333af67312ba0de5018ef1c9ba663cb18549443e568f0bd35262dc1c48'),
    'tokenizer.json': (466247, 'be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037'),
    'model.safetensors': (133466304, 'd2d541e5f101695ae495eacd867a8d025ecfe8f9674fb23aa6cf93cdb60a5542'),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-assets', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    save(a.output/'protocol.json', dict(batch='A113-assets', model=MODEL, revision=REVISION,
         required_files=FILES, runner_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT/'docs/batches/A112_A114_native_and_H1_queue_plan.md'),
         automatic_retry=False, request_timeout=60, download_window_seconds=1800,
         actual_classifier_fits=0, real_API_requests=0))
    assets = []
    try:
        folder = a.output/'minilm_l12'
        folder.mkdir()
        with requests.Session() as session:
            session.proxies.update(http='http://127.0.0.1:7897', https='http://127.0.0.1:7897')
            for name, (size, digest) in FILES.items():
                path = folder/name
                if name == 'tokenizer.json':
                    source = a.base_assets/'minilm'/name
                    if source.stat().st_size != size or sha(source) != digest:
                        raise ValueError('Local shared tokenizer differs')
                    shutil.copyfile(source, path)
                else:
                    partial = path.with_name(name+'.partial')
                    offset = 0
                    while offset < size:
                        if time.monotonic()-started > 1800:
                            raise TimeoutError('Public asset acquisition window elapsed')
                        end = min(size-1, offset+4*1024*1024-1)
                        url = f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}?download=true&segment={offset}'
                        with session.get(url, headers={'Range': f'bytes={offset}-{end}', 'Accept-Encoding': 'identity'},
                                         stream=True, timeout=60) as response:
                            response.raise_for_status()
                            if response.status_code == 200 and offset == 0:
                                limit = size
                            elif response.status_code == 206 and response.headers.get('Content-Range') == f'bytes {offset}-{end}/{size}':
                                limit = end-offset+1
                            else:
                                raise ValueError('Unexpected asset range response')
                            count = 0
                            chunks = []
                            for chunk in response.iter_content(64*1024):
                                if time.monotonic()-started > 1800:
                                    raise TimeoutError('Public asset acquisition window elapsed')
                                count += len(chunk)
                                if count > limit:
                                    raise ValueError('Asset response exceeds declared bytes')
                                chunks.append(chunk)
                            if count != limit:
                                raise ValueError('Incomplete asset segment; no retry')
                        with partial.open('xb' if offset == 0 else 'ab') as target:
                            for chunk in chunks:
                                target.write(chunk)
                        offset += count
                        save(a.output/'download_progress.json', dict(file=name, bytes=offset, expected_bytes=size,
                             elapsed_seconds=time.monotonic()-started))
                        print(dict(file=name, bytes=offset, expected=size), flush=True)
                    if partial.stat().st_size != size or sha(partial) != digest:
                        raise ValueError('Pinned B asset checksum differs: '+name)
                    partial.rename(path)
                assets.append(dict(name=name, bytes=size, sha256=digest, verified=True))
                save(a.output/'assets.json', assets)
        save(a.output/'completion.json', dict(status='ready', model=MODEL, revision=REVISION, verified_files=3,
             actual_classifier_fits=0, real_API_requests=0, elapsed_seconds=time.monotonic()-started))
    except Exception as error:
        save(a.output/'completion.json', dict(status='stopped', error=str(error), automatic_retry=False,
             actual_classifier_fits=0, real_API_requests=0, elapsed_seconds=time.monotonic()-started))
        raise


if __name__ == '__main__':
    main()
