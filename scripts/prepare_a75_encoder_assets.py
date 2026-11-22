"""Prepare an isolated overlay environment and hash-locked public encoder assets; no school API."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import requests
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import sha
from superagent.ab_sentence import FILES, MODEL, REVISION
from superagent.ab_vision import WEIGHT_BYTES, WEIGHT_SHA256

ROOT=Path(__file__).resolve().parents[1]
PINS=['numpy==2.3.4','scipy==1.16.3','scikit-learn==1.7.2','joblib==1.5.2','requests==2.34.2',
      'Pillow==12.3.0','catboost==1.2.10','transformers==4.53.3','tokenizers==0.21.4',
      'safetensors==0.6.2','huggingface-hub==0.34.4']


def main(a):
    root=a.output.resolve();root.mkdir(parents=True,exist_ok=False);began=time.monotonic();assets=[]
    save(root/'protocol.json',dict(batch=a.batch,runner_sha256=sha(Path(__file__)),public_asset_process_proxy=a.public_proxy,
        plan_sha256=sha(ROOT/a.plan),base_python=str(a.base_python),
        overlay_system_site_packages=True,pinned_overlay=PINS,actual_classifier_fits=0,real_api_requests=0,
        credential_used=False,automatic_retry=False,timeout_seconds=60))
    try:
        with (root/'create.log').open('w',encoding='utf-8') as log:
            subprocess.run([str(a.base_python),'-m','venv','--system-site-packages',str(root/'venv')],
                           stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
        python=root/'venv/Scripts/python.exe'
        proxy_args=['--proxy',a.public_proxy] if a.public_proxy else []
        with (root/'install.log').open('w',encoding='utf-8') as log:
            subprocess.run([str(python),'-m','pip','install','--index-url','https://pypi.org/simple','--disable-pip-version-check',
                '--retries','0','--timeout','60',*proxy_args,'--only-binary=:all:','--ignore-installed','--no-deps',*PINS],
                stdout=log,stderr=subprocess.STDOUT,check=True,timeout=900)
        program="""import json,sys
from importlib.metadata import version
import torch,torchvision,transformers,tokenizers,safetensors,sklearn,numpy,scipy,joblib,requests,PIL,catboost
names=['torch','torchvision','transformers','tokenizers','safetensors','scikit-learn','numpy','scipy','joblib','requests','Pillow','catboost','huggingface-hub']
print(json.dumps(dict(python=sys.version,executable=sys.executable,versions={p:version(p) for p in names},torch_source=torch.__file__,sklearn_source=sklearn.__file__)))"""
        result=subprocess.run([str(python),'-c',program],capture_output=True,text=True,encoding='utf-8',timeout=120)
        (root/'import.log').write_text(result.stderr,encoding='utf-8')
        if result.returncode:raise RuntimeError('Overlay import check failed; see import.log')
        environment=json.loads(result.stdout);save(root/'environment.json',environment)
        if not environment['versions']['torch'].startswith('2.13.0') or not environment['versions']['torchvision'].startswith('0.28.0'):
            raise ValueError('Inherited torch/vision versions differ')
        for pin in PINS:
            name,value=pin.split('==')
            if environment['versions'][name]!=value:raise ValueError('Overlay version differs: '+name)
        for name,(size,digest) in FILES.items():
            assets.append((root/'minilm'/name,f'https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}',size,digest))
        assets.append((root/'mobilenet_v3_small-047dcff4.pth','https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',WEIGHT_BYTES,WEIGHT_SHA256))
        records=[]
        with requests.Session() as session:
            session.trust_env=True
            if a.public_proxy:session.proxies.update(http=a.public_proxy,https=a.public_proxy)
            for path,url,size,digest in assets:
                path.parent.mkdir(parents=True,exist_ok=True);partial=path.with_name(path.name+'.partial');checksum=hashlib.sha256();downloaded=0
                with session.get(url,timeout=60,stream=True,headers={'User-Agent':'SuperAgent-research-asset-verifier'}) as response:
                    response.raise_for_status()
                    with partial.open('xb') as stream:
                        for chunk in response.iter_content(1024*1024):
                            stream.write(chunk);checksum.update(chunk);downloaded+=len(chunk)
                            if downloaded>size:raise ValueError('Asset exceeds pinned bytes: '+path.name)
                if downloaded!=size or checksum.hexdigest()!=digest:raise ValueError('Pinned asset mismatch: '+path.name)
                partial.rename(path);records.append(dict(file=str(path),source=url,bytes=downloaded,sha256=digest,verified=True));save(root/'assets.json',records)
        save(root/'completion.json',dict(status='ready',actual_classifier_fits=0,real_api_requests=0,verified_assets=len(records),
            environment_python=str(python),elapsed_seconds=time.monotonic()-began,model_training_not_executed=True))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),automatic_retry=False,actual_classifier_fits=0,
            real_api_requests=0,elapsed_seconds=time.monotonic()-began));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-python',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',default='A75');p.add_argument('--plan',default='docs/batches/A75_encoder_environment_plan.md')
    p.add_argument('--public-proxy',choices=['http://127.0.0.1:7897'])
    main(p.parse_args())
