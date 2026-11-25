"""Fetch the four frozen public feature encoders, or verify already-downloaded weights."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(root, check_only):
    manifest = json.loads((Path(__file__).parent/'verification/weights.json').read_text(encoding='utf-8'))
    for repo, record in manifest.items():
        folder = root/repo.split('/')[-1]
        for name, expected in record['sha256'].items():
            path = folder/name
            if not path.is_file() and not check_only:
                from huggingface_hub import hf_hub_download
                hf_hub_download(repo, name, revision=record['revision'], local_dir=folder)
            if not path.is_file():
                raise FileNotFoundError(path)
            with path.open('rb') as file:
                if hashlib.file_digest(file, 'sha256').hexdigest() != expected:
                    raise ValueError(f'Encoder file differs from the verified release: {path}')
        print(f'{repo} verified')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('weights', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    prepare(args.weights, args.check_only)
