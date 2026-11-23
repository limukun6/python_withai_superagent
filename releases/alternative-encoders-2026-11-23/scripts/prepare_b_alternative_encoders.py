"""Fetch public data-only files into local weight directories and verify publisher identities."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import requests


REGISTERED={
    'e5':('intfloat/e5-small-v2','ffb93f3bd4047442299a41ebb6fa998a38507c52',
          (615,'c5aba64a4d8f4c631ef0beddc89704232d2e649d'),
          '45bfa60070649aae2244fbc9d508537779b93b6f353c17b0f95ceccb1c5116c1'),
    'bge':('BAAI/bge-small-en-v1.5','5c38ec7c405ec4b44b94cc5a9bb96e735b38267a',
           (743,'3992bf890728a92476c700e6e657fd696eb94160'),
           '3c9f31665447c8911517620762200d2245a2518d6e7208acc78cd9db317e21ad')}


def identity(path,expected_size,expected):
    if path.stat().st_size!=expected_size:
        raise ValueError('Unexpected file size')
    with path.open('rb') as f:
        sha256=hashlib.file_digest(f,'sha256').hexdigest()
    if len(expected)==40:
        actual=hashlib.sha1(f'blob {expected_size}\0'.encode()+path.read_bytes()).hexdigest()
    else:
        actual=sha256
    if actual!=expected:
        raise ValueError('File identity does not match')
    return sha256


def fetch(name,directory):
    repo,revision,config,weights=REGISTERED[name]
    dest=directory/repo.split('/')[-1]
    dest.mkdir(parents=True,exist_ok=True)
    rec={}
    for file,(size,expected) in {'config.json':config,'tokenizer.json':(711396,'688882a79f44442ddc1f60d70334a7ff5df0fb47'),
                                 'model.safetensors':(133466304,weights)}.items():
        target=dest/file
        if target.exists():
            rec[file]=dict(bytes=size,sha256=identity(target,size,expected),source='verified existing file')
            continue
        urls=[('HF mirror',f'https://hf-mirror.com/{repo}/resolve/{revision}/{file}'),
              ('ModelScope',f'https://modelscope.cn/models/{repo}/resolve/master/{file}')]
        for provider,url in urls:
            with tempfile.NamedTemporaryFile(dir=dest,delete=False) as fh:
                part=Path(fh.name)
            try:
                with requests.get(url,stream=True,timeout=(10,35)) as res:
                    res.raise_for_status()
                    with part.open('wb') as f:
                        for block in res.iter_content(1024*1024):
                            f.write(block)
                            if f.tell()>size:
                                raise ValueError('File exceeds the registered length')
                actual=identity(part,size,expected)
                os.replace(part,target)
                rec[file]=dict(bytes=size,sha256=actual,source=provider)
                print(name,file,provider,flush=True)
                break
            except (requests.RequestException,ValueError) as exc:
                print(name,file,provider,type(exc).__name__,flush=True)
            finally:
                part.unlink(missing_ok=True)
        else:
            raise RuntimeError('Verified asset unavailable: '+file)
    print(json.dumps(dict(model=repo,revision=revision,files=rec),indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('name',choices=REGISTERED)
    p.add_argument('directory',type=Path)
    a=p.parse_args()
    fetch(a.name,a.directory)
