"""No-fit inference throughput on known training text, with full-corpus tokenizer workload."""
import argparse
import os
from pathlib import Path
import sys
import time
import numpy as np
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--assets',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    os.environ.update(SUPERAGENT_MINILM_WEIGHTS=str(a.assets/'minilm'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    from scripts import a38_predict_guard
    from superagent.ab_sentence import encode,_encoder,verified_directory
    from superagent.ab_run import sources
    from superagent.data import load_training_task
    frozen=sources()
    save(out/'protocol.json',dict(batch='A87',runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A87_encoder_runtime_calibration_plan.md'),source_files=frozen,
        sample_sizes=[128,512,1024],actual_classifier_fits=0,real_api_requests=0,training_only=True))
    results={}
    try:
        entries=read(a.reference/'protocol.json')['entries']
        for name in ('sms','preference'):
            entry=entries[name];src=Path(entry['training_source'])
            if sha(src)!=entry['training_source_sha256']:raise ValueError('Training input changed')
            task=load_training_task(src);texts=list(dict.fromkeys([s for pair in task.X_train for s in pair] if task.kind=='pair_text' else task.X_train))
            if len(texts)<1024:raise ValueError('Insufficient registered unique training text')
            warm=time.monotonic();encode(texts[:64]);warm_seconds=time.monotonic()-warm
            tokenizer,_=_encoder(verified_directory());full_tokens=sum(len(x.ids) for x in tokenizer.encode_batch(texts));measurements=[]
            for size in (128,512,1024):
                began=time.monotonic();vectors,tokens=encode(texts[:size]);seconds=time.monotonic()-began
                if vectors.shape!=(size,384) or not np.isfinite(vectors).all():raise ValueError('Encoder features invalid')
                measurements.append(dict(unique_texts=size,input_tokens=tokens,seconds=seconds,tokens_per_second=tokens/seconds,
                    estimated_full_unique_transform_seconds=full_tokens*seconds/tokens))
                save(out/'partial.json',dict(completed_tasks=results,current_task=name,measurements=measurements))
            estimates=[r['estimated_full_unique_transform_seconds'] for r in measurements]
            results[name]=dict(training_rows=len(task.y_train),full_unique_texts=len(texts),full_unique_truncated_tokens=full_tokens,
                warmup_seconds=warm_seconds,measurements=measurements,estimated_one_transform_seconds_range=[min(estimates),max(estimates)],
                estimate_not_complete_worker_guarantee=True)
            save(out/'results.json',results)
        save(out/'completion.json',dict(status='passed',actual_classifier_fits=0,real_api_requests=0,A_cumulative_fits_after=1247,
            source_unchanged=sources()==frozen,elapsed_seconds=time.monotonic()-start,formal_test_or_H_opened=False))
    except Exception as error:
        save(out/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,real_api_requests=0,
            automatic_retry=False,elapsed_seconds=time.monotonic()-start));raise
