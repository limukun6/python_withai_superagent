"""Read registered task metadata only and snapshot the actually executable shared catalogue."""
import argparse
import os
from pathlib import Path
import time
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--encoder-policy',choices=['restricted','research'],default='restricted');a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    os.environ.update(SUPERAGENT_MINILM_WEIGHTS=str(a.assets/'minilm') if a.encoder_policy=='research' else '',
        SUPERAGENT_MOBILENET_V3_SMALL_WEIGHTS=str(a.assets/'mobilenet_v3_small-047dcff4.pth') if a.encoder_policy=='research' else '')
    from scripts import a38_predict_guard
    from superagent.ab_run import sources,registered_capabilities
    from superagent.scheduler_bridge import fingerprint
    baseline=read(ROOT/'research/a73_b_upload_review/contracts_final/local_inventory.json')['tasks']
    entries=read(a.reference/'protocol.json')['entries'];results={}
    save(out/'protocol.json',dict(batch='A83_catalogue',runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A83_shared_runtime_handoff_plan.md'),source_files=sources(),
        metadata_only=True,encoder_policy=a.encoder_policy,encoder_course_exception_verified=False,
        training_rows_source='A73 measured A47 training registration',real_api_requests=0,actual_classifier_fits=0))
    for name,entry in entries.items():
        spec=entry['task_spec_current'];rows=baseline[name]['full_training_rows']
        full=registered_capabilities(spec,rows,'full');old=registered_capabilities(spec,rows,'baseline')
        results[name]=dict(full_training_rows=rows,primary_metric=spec['metric'],task_spec=spec,
            full_capabilities=full,baseline_capabilities=old,full_candidates=len(full),baseline_candidates=len(old),
            full_menu_sha256=fingerprint(full),baseline_menu_sha256=fingerprint(old),training_source_sha256=entry['training_source_sha256'],
            actual_encoder_candidates=[k for k,v in full.items() if v['resource']=='local_frozen_encoder_cpu'])
    save(out/'results.json',results)
    save(out/'completion.json',dict(status='passed',actual_classifier_fits=0,real_api_requests=0,
        formal_protocol_frozen=False,H_or_test_opened=False,elapsed_seconds=time.monotonic()-start))
    print({name:dict(full=row['full_candidates'],baseline=row['baseline_candidates'],encoders=row['actual_encoder_candidates']) for name,row in results.items()})
