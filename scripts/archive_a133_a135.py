"""Archive explicit metadata receipts; keep data, models, weights and ledgers private."""
import argparse
import json
from pathlib import Path
import shutil
from scripts.a_batch_common import hashes,save
from scripts.run_a10_real_task_pilot import read,sha

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    batches={
      'a133_intake':('a133_B_routing_intake',['protocol.json','results.json','completion.json']),
      'a133_first':('a133_routing_verification',['protocol.json','results.json','completion.json','targeted_tests.log']),
      'a133':('a133_routing_verification_v2',['protocol.json','results.json','completion.json','targeted_tests.log',
                 'verify_b_encoder_cascade.log','verify_b_native_encoder_routing.log']),
      'a134':('a134_routed_delivery',['protocol.json','results.json','completion.json']),
      'a135_prepare_failed':('a135_task1_current',['protocol.json','completion.json']),
      'a135':('a135_task1_current_v2',['protocol.json','prepared.json','model_frozen.json','source_access.json',
              'command.json','results.json','completion.json','physical_fits.jsonl','verification.json',
              'run/run_manifest.json','run/task_spec.json','run/final_selection.json','run/run_outcome.json',
              'run/actions.json','run/classifier_fits.json','run/history.json'])}
    records={}
    for name,(folder,names) in batches.items():
        src=BASE/'runs'/folder;dst=out/name;dst.mkdir()
        for n in names:
            target=dst/n;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src/n,target)
        records[name]=dict(original_folder=str(src),files={n:sha(dst/n) for n in names})
    current=hashes(ROOT);manifest=read(out/'a135/run/run_manifest.json')['source_files']
    if current!={'superagent/'+k:v for k,v in manifest.items()}:raise ValueError('Source differs from model source')
    for name in ('a133','a134','a135'):
        protocol=read(out/name/'protocol.json');end=read(out/name/'completion.json')
        if protocol['source_files']!=current or end['status']!='passed':raise ValueError('Source/completion drift: '+name)
        if end['protocol_sha256']!=sha(out/name/'protocol.json') or end['results_sha256']!=sha(out/name/'results.json'):
            raise ValueError('Result seal differs: '+name)
    owner=read(ROOT/'research/a130_a132_H1_summary/current_ledger_owner.json')
    if sha(Path(owner['ledger_path']))!=owner['ledger_sha256']:raise ValueError('Ledger changed during zero-API batch')
    save(out/'current_ledger_owner.json',owner)
    save(out/'summary.json',dict(status='passed',A_cumulative_fits_before=1588,A_cumulative_fits_after=1599,
         new_classifier_fits=11,new_model_API_calls=0,new_API_cost_rmb=0,
         source_files=current,provider='924e88ab5f6b49d90d1baf2b835f533e644a5776',
         routing_default_enabled=False,task1_final_submission_frozen=False,
         ledger_unchanged=True,B_latest_registered_only='5bd6b145e63ca59af005aab6cf1e9cd24ff21931',records=records))
    save(out/'completion.json',dict(status='archived',summary_sha256=sha(out/'summary.json'),
         actual_classifier_fits=0,real_API_requests=0,model_data_weights_ledger_copied=False))
    print(json.dumps(dict(status='archived',new_classifier_fits=11,new_model_API_calls=0,A_cumulative_fits=1599)))


if __name__=='__main__':main()
