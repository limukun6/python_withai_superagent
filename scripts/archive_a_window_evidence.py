"""Archive compact, allowlisted evidence; exclude datasets, credentials and models."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from a_batch_common import save

BATCHES=('a05_baseline_dev','a05_fixed_dev','a05_baseline_reserved','a05_fixed_reserved','a05_final_regression',
    'a06_baseline','a06_fixed','a06_final_regression','a07_prepared','a07_delivery','a08_gate','a08_gate_final',
    'a_window_regression','a_window_cli_final','a07_export_baseline','a07_export_fixed','a07_final_export')


def main():
    p=argparse.ArgumentParser();p.add_argument('--runs-root',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False);inventory=[]
    for batch in BATCHES:
        source=Path(a.runs_root)/batch;target=output/batch;target.mkdir()
        files=[source/n for n in ('protocol.json','results.json','completion.json','tests.log','worker_calls.json','tasks.json') if (source/n).is_file()]
        if batch=='a07_delivery':
            files.extend(source/arm/'export_verification.json' for arm in ('bean_baseline','bean_fixed','sms_baseline','sms_fixed','digits_fixed'))
        if batch=='a08_gate_final':files.append(source/'events.jsonl')
        if batch=='a_window_cli_final':files.extend(source.glob('cases/*/stats.json'))
        for file in files:
            relative=file.relative_to(source);destination=target/relative;destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(file,destination)
            inventory.append({'path':destination.relative_to(output).as_posix(),'sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'bytes':destination.stat().st_size})
    save(output/'inventory.json',inventory)
    print(json.dumps({'files':len(inventory),'bytes':sum(x['bytes'] for x in inventory)}))


if __name__=='__main__':main()
