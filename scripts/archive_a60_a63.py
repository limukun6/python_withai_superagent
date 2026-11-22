"""Archive compact A60-A63 evidence; no data, models, score vectors, credentials or ledgers."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);index=[]
    sets={
        'a60_matched_finalists':['protocol.json','completion.json','results.json','physical_fits.jsonl'],
        'a61_auc_readiness':['protocol.json','completion.json','artificial_results.json','real_model_diagnostics.json'],
        'a62_auc_official_loop':['protocol.json','completion.json','results.json','physical_fits.jsonl'],
        'a63_auc_recovery_delivery':['protocol.json','completion.json','fault_results.json','delivery_results.json'],
        'a63_clean_environment':['protocol.json','completion.json','results.json'],
    }
    for name,files in sets.items():
        folder=a.output/name;folder.mkdir()
        for file in files:
            source=a.runs/name/file;target=folder/file;shutil.copy2(source,target)
            value=[json.loads(v) for v in source.read_text(encoding='utf-8').splitlines()] if file.endswith('.jsonl') else read(source)
            canonical=hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
            index.append(dict(path=target.relative_to(a.output).as_posix(),original_runtime_bytes_sha256=sha(source),canonical_value_sha256=canonical))
    # Counts derive from physical fit observations, not number of completed summaries.
    count=lambda name:len((a.runs/name/'physical_fits.jsonl').read_text(encoding='utf-8').splitlines())
    n60,n62=count('a60_matched_finalists'),count('a62_auc_official_loop')
    if n60!=26 or n62!=3:raise ValueError('Unexpected actual fit count; reconcile before publishing')
    save(a.output/'resource_audit.json',dict(starting_tracked_A_fits=970,new_fits_by_batch={'A60':n60,'A61':0,'A62':n62,'A63':0},
        actual_new_classifier_fits=n60+n62,tracked_A_fits_after=970+n60+n62,real_api_requests=0,
        additional_API_course_cost_rmb=0,historical_unknown_request_count=11,formal_test_or_H_opened=False,
        overall_course_budget_compliance='unverified_history',new_training_stopped_near_original_suggested_ceiling=True))
    save(a.output/'index.json',dict(files=index,normalization='Git text LF normalization may change bytes; canonical JSON hashes verify values.',
        models_datasets_probabilities_credentials_ledgers_copied=False))
