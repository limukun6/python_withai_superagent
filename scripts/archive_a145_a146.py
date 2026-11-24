"""Archive only named evidence, scanning for credential bytes before copying."""
import argparse
import json
from pathlib import Path

from scripts.a_batch_common import hashes,save
from scripts.run_a10_real_task_pilot import read,sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('cases','verification','retained','freeze','failure','inputs','run','evaluation','output','api-key-file'):
        p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);files={}
    secret=a.api_key_file.read_text(encoding='utf-8').strip().encode()
    def copy(path,relative):
        raw=path.read_bytes()
        if secret and secret in raw:raise ValueError('Credential detected; no content printed')
        target=out/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);files[relative]=sha(target)
    for group,folder in (('cases',a.cases),('verification',a.verification),('retained',a.retained),
                         ('initial_failure',a.failure),('inputs',a.inputs),('paid',a.run),('evaluation',a.evaluation)):
        # Whitelist metadata; no SQLite, models, feature arrays, data packages or keys.
        for file in folder.rglob('*'):
            rel=file.relative_to(folder)
            if file.is_file() and 'private' not in rel.parts and file.suffix in ('.json','.jsonl','.log'):
                copy(file,group+'/'+rel.as_posix())
    copy(a.freeze,'production_freeze.json')
    if read(a.freeze)['source_files']!=hashes(ROOT):raise ValueError('Frozen production source changed')
    result=read(a.run/'results.json');complete=read(a.run/'completion.json')
    save(out/'archive_receipt.json',dict(status='archived',source_files=hashes(ROOT),files_sha256=files,
        paid_calls_referenced=complete['real_API_requests'],actual_classifier_fits=0,real_API_requests=0,
        historical_rows_unchanged=result['historical_rows_unchanged'],ledger_or_key_copied=False,
        paid_completion_sha256=sha(a.run/'completion.json'),evaluation_completion_sha256=sha(a.evaluation/'completion.json')))
    print(json.dumps(dict(status='archived',files=len(files),ledger_or_key_copied=False)))


if __name__=='__main__':main()
