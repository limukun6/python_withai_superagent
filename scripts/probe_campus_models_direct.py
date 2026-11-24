"""One authenticated read-only model-list query with explicit campus resolution; no chat or retry."""
import argparse
from pathlib import Path
import time

import requests

from scripts.a_batch_common import save
from scripts.campus_direct import campus_resolution,CAMPUS_IP,HOST


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--api-key-file',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);began=time.monotonic()
    save(args.output/'protocol.json',dict(endpoint='https://'+HOST+'/api/v2/models',explicit_process_dns=CAMPUS_IP,
        TLS_certificate_verification=True,proxy_configuration_modified=False,real_chat_requests=0,automatic_retry=False))
    try:
        with campus_resolution(CAMPUS_IP),requests.Session() as session:
            session.trust_env=False
            response=session.get('https://'+HOST+'/api/v2/models',headers={'Authorization':'Bearer '+args.api_key_file.read_text(encoding='utf-8').strip()},timeout=(10,60))
            response.raise_for_status();data=response.json()
        entries=data.get('data',[]) if isinstance(data,dict) else data
        ids=[row.get('id') for row in entries if isinstance(row,dict)]
        if 'DeepSeek-V4.1-Flash' not in ids:raise ValueError('Expected Flash model not present in the actual JSON model list')
        record=dict(status='passed',http_status=response.status_code,model_count=len(ids),flash_available=True,
            real_chat_requests=0,automatic_retry=False,elapsed_seconds=time.monotonic()-began)
        save(args.output/'completion.json',record);print(record)
    except Exception as error:
        save(args.output/'completion.json',dict(status='stopped',error_type=type(error).__name__,error=str(error),
            real_chat_requests=0,automatic_retry=False,elapsed_seconds=time.monotonic()-began));raise
