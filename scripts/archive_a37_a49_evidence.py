"""Archive only explicit compact evidence; never models, inputs, credentials, or ledgers."""
import argparse
from pathlib import Path
import shutil

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);index=[]
    files={
        'a37_training_profile_probe_final':['protocol.json','completion.json','results.json'],
        'a39_partial_delivery_prototype_final':['protocol.json','completion.json','final_selection.json'],
        'a40_interruption_audit':['protocol.json','completion.json','results.json'],
        'a40_partial_coupon':['protocol.json','completion.json','final_selection.json'],
        'a41_guard_success':['protocol.json','completion.json'],
        'a41_guard_expected_failure':['protocol.json','completion.json'],
        'a42_recovery_supplement':['protocol.json','completion.json'],
        'a42_dns_no_dispatch':['proof.json','completion.json'],
        'a44_regression':['protocol.json','completion.json','fit_observations.json'],
        'a44_a46_offline':['protocol.json','completion.json'],
        'a44_a46_offline_fixed':['protocol.json','completion.json','results.json'],
        'a44_a46_offline_final':['protocol.json','completion.json','results.json','physical_fits.jsonl'],
        'a44_context_audit':['protocol.json','completion.json','results.json'],
        'a46_checkpoint_probes':['protocol.json','completion.json','results.json'],
        'a48_calibrated_mock_loop':['protocol.json','completion.json','results.json','physical_fits.jsonl'],
        'a49_parameter_probe':['protocol.json','completion.json','results.json'],
        'a50_models_direct':['protocol.json','completion.json'],
    }
    for name,names in files.items():
        target=root/name;target.mkdir()
        for file in names:
            source=args.runs/name/file
            if not source.is_file():
                if file=='protocol.json' and name=='a44_regression':continue
                raise FileNotFoundError(source)
            shutil.copy2(source,target/file)
            index.append(dict(run=name,file=file,sha256=sha(source)))
    save(root/'index.json',dict(files=index,models_or_data_copied=False,API_keys_copied=False,ledgers_copied=False,
        scope='Snapshots at each recorded execution time; later live ledger use is separate'))
