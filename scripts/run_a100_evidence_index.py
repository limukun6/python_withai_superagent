"""Index reusable archived evidence and matched C1 contrasts without retraining."""
import argparse
from pathlib import Path
import time
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read, sha
from superagent.ab_run import sources

ROOT = Path(__file__).resolve().parents[1]

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    save(a.output/'protocol.json', dict(batch='A100', runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A100_A101_evidence_freeze_plan.md'),
        source_files=sources(), actual_classifier_fits=0, real_API_requests=0))
    try:
        records = []; contrasts = []; references = {}
        previous = None
        for batch, task in [('a89_coupon_C1','coupon'), ('a97_flower_C1','flowers')]:
            folder = ROOT/'research'/batch
            protocol = read(folder/'protocol.json'); summary = read(folder/'summary.json')
            if summary['completion']['status'] != 'completed': raise ValueError('Incomplete C1')
            if previous is not None and previous != protocol['source_files']: raise ValueError('Cannot pool different C1 source')
            previous = protocol['source_files']
            for name in ('protocol.json','summary.json','completion.json'):
                path = folder/name; references[path.relative_to(ROOT).as_posix()] = sha(path)
            records.append(dict(task=task, role='matched_full_C1', evidence=batch,
                arms=6, completed=True, actual_new_fits=summary['completion']['actual_classifier_fits'],
                reused_fits=summary['completion']['reused_classifier_fits'],
                actual_API_requests=summary['completion']['real_api_requests'],
                known_incremental_estimated_rmb=summary['known_incremental_estimated_rmb']))
            for c in summary['contrasts']:
                contrasts.append(dict(task=task, **c))
        for task, relative in [('preference','b1_07_allocation'), ('bean','b1_08_allocation/bean'),
                               ('coupon','b1_08_allocation/coupon'), ('sms','b1_08_allocation/sms')]:
            folder = ROOT/'research'/relative
            completion = read(folder/'completion.json'); protocol = read(folder/'protocol.json')
            result = read(folder/'results.json')
            if completion['status'] != 'passed' or len(result['paths']) != 4: raise ValueError('B historical evidence incomplete')
            for name in ('protocol.json','results.json','completion.json'):
                path=folder/name; references[path.relative_to(ROOT).as_posix()]=sha(path)
            records.append(dict(task=task, role='historical_local_fixed_dynamic', evidence=relative,
                paths=4, reported_fits=completion['fits'], reported_API_requests=completion['api_requests'],
                source_sha256=protocol['source_sha256'],
                source_input_rows={k:v['inner_input_rows'] for k,v in protocol['tasks'].items()},
                reuse='Keep original development comparison; not a replacement for current six-arm C1; do not add fits to A'))
        q = ROOT/'research/bc1_05_scoring/results.json'
        references[q.relative_to(ROOT).as_posix()] = sha(q)
        records.append(dict(role='historical_Q_and_reload', scores=read(q)['scores'],
            reuse='Keep method availability and original score evidence; different data/menu/source from A94'))
        current = sources()
        save(a.output/'results.json', dict(records=records, C1_contrasts=contrasts, evidence_sha256=references,
            production_changes_since_C1=[k for k in sorted(set(current)|set(previous)) if current.get(k)!=previous.get(k)],
            current_gaps=dict(current_six_arm_tasks_not_documented=['preference','bean','sms'],
                independent_semantic_inputs='48 cases and 24 annotations not received; no confirmed teammate assignment',
                H_performance='sealed; no scores', final_version_C1='current prose changes have no new end-to-end performance results'),
            decision='Reuse existing B local/Q evidence; current full C1 has two tasks; no automatic full rerun assignment'))
        save(a.output/'completion.json', dict(status='passed', indexed_records=len(records), indexed_files=len(references),
            actual_classifier_fits=0, real_API_requests=0, elapsed_seconds=time.monotonic()-start,
            A_cumulative_fits_after=1438, formal_test_or_H_opened=False))
        print(dict(records=len(records), files=len(references), matched_C1_tasks=2,
            source_changes=[k for k in current if current[k]!=previous.get(k)]))
    except Exception as e:
        save(a.output/'completion.json', dict(status='stopped', error=str(e), actual_classifier_fits=0,
            real_API_requests=0, elapsed_seconds=time.monotonic()-start)); raise
