"""Download pinned E5 public files with checked ranges and zero automatic retries."""
import argparse
import hashlib
from pathlib import Path
import time
import requests
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import sha
from superagent.ab_sentence import ALTERNATIVES

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();spec=ALTERNATIVES['e5'];requests_count=0
    save(a.output/'protocol.json',dict(batch='A126-E5-assets',model=spec['model'],revision=spec['revision'],
        files=spec['files'],runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A126_E5_assets_plan_20261123_zh.md'),maximum_classifier_fits=0,
        real_API_requests=0,automatic_retry=False,request_timeout=60,window_seconds=1800))
    try:
        folder=a.output/'e5-small-v2';folder.mkdir();records=[]
        with requests.Session() as session:
            session.proxies.update(http='http://127.0.0.1:7897',https='http://127.0.0.1:7897')
            for name,(size,digest) in spec['files'].items():
                path=folder/name;partial=folder/(name+'.partial');offset=0
                while offset<size:
                    if time.monotonic()-start>1800:raise TimeoutError('Public download window elapsed')
                    end=min(size-1,offset+4*1024*1024-1)
                    url=f'https://huggingface.co/{spec["model"]}/resolve/{spec["revision"]}/{name}?download=true&segment={offset}'
                    requests_count+=1
                    with session.get(url,headers={'Range':f'bytes={offset}-{end}','Accept-Encoding':'identity'},stream=True,timeout=60) as response:
                        response.raise_for_status()
                        if response.status_code==200 and offset==0:limit=size
                        elif response.status_code==206 and response.headers.get('Content-Range')==f'bytes {offset}-{end}/{size}':limit=end-offset+1
                        else:raise ValueError('Unexpected public range response')
                        count=0
                        with partial.open('xb' if offset==0 else 'ab') as stream:
                            for chunk in response.iter_content(65536):
                                if time.monotonic()-start>1800:raise TimeoutError('Public download window elapsed')
                                count+=len(chunk)
                                if count>limit:raise ValueError('Range exceeds registered size')
                                stream.write(chunk)
                        if count!=limit:raise ValueError('Incomplete public segment; stopped without retry')
                    offset+=count
                    save(a.output/'progress.json',dict(file=name,bytes=offset,expected=size,public_requests=requests_count))
                    print(dict(file=name,bytes=offset,total=size),flush=True)
                if partial.stat().st_size!=size or sha(partial)!=digest:raise ValueError('Public bytes differ from B pin: '+name)
                partial.rename(path);records.append(dict(name=name,bytes=size,sha256=digest))
        save(a.output/'assets.json',records)
        save(a.output/'completion.json',dict(status='ready',actual_classifier_fits=0,real_API_requests=0,
            public_download_requests=requests_count,elapsed_seconds=time.monotonic()-start))
    except Exception as error:
        save(a.output/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,real_API_requests=0,
            public_download_requests=requests_count,automatic_retry=False,elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':main()
