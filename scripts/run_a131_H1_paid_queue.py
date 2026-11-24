"""Resume two H1 R0 paths on the original cumulative ledger with an explicit accepted-history snapshot."""
import argparse
import json
from pathlib import Path
import sys
import time
from scripts.a_batch_common import save,hashes
from scripts.run_a10_real_task_pilot import read,sha
from scripts.run_a122_current_agent_acceptance import environment,child,physical
from tools.accepted_usage_snapshot import accepted_history

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('snapshot','authorization','ledger','api-key-file','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--validate-only',action='store_true');a=p.parse_args();a.output=a.output.resolve()
    a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();source=hashes(ROOT)
    from superagent.task_budget import TaskBudgetLedger
    from superagent.research_budget import research_preflight
    ledger=TaskBudgetLedger(a.ledger);before=ledger.project_summary();authorization_sha=sha(a.authorization)
    save(a.output/'protocol.json',dict(batch='A131',source_files=source,runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A130_A132_H1_paid_continuation_plan_20261123_zh.md'),
        accepted_history_sha256=authorization_sha,project_usage_before=before,ledger_sha256_before=sha(a.ledger),
        maximum_classifier_fits=0 if a.validate_only else 22,maximum_new_API_requests=0 if a.validate_only else 6,
        run_order=['61411','61421'],estimated_stop_allowance_rmb=.60,starting_A_fits=1566,automatic_retry=False))
    results=[]
    def fits():return sum(len(physical(a.output/s)) for s in ('61411','61421'))
    try:
        if not a.api_key_file.is_file():raise FileNotFoundError('Configured local key file missing')
        for seed in ('61411','61421'):
            current=ledger.project_summary()
            with accepted_history(a.authorization,authorization_sha):research_preflight(ledger,'h1_transfer',.30)
            if current['requests']-before['requests']>6:raise ValueError('Batch API request ceiling exceeded')
            owner=a.output/(seed+'_writer.json')
            save(owner,dict(owner='A',window_status='exclusive_active',ledger_sha256=sha(a.ledger),
                project_summary=current,accepted_history_sha256=authorization_sha,
                authority='User authorized continuation; B already stopped writing'))
            command=[sys.executable,'-m','scripts.run_a123_H1_frozen','train','--prepared',str(a.snapshot),
                '--outer',seed,'--arm','R0','--output',str(a.output/seed),'--ledger',str(a.ledger),
                '--api-key-file',str(a.api_key_file),'--writer-receipt',str(owner),'--writer-sha256',sha(owner),
                '--accepted-history',str(a.authorization),'--accepted-history-sha256',authorization_sha]
            if a.validate_only:command+=['--validate-only']
            env=environment(a.output);env['SUPERAGENT_OFFLINE']='0'
            process=child(command,a.output/(seed+'.log'),env,1380,allow_failure=True)
            done=read(a.output/seed/'completion.json')
            results.append(dict(outer=seed,arm='R0',process=process,completion=done))
            save(a.output/'checkpoint.json',results)
            # This checks again after the child: any new unknown/pending stops the queue.
            with accepted_history(a.authorization,authorization_sha):research_preflight(ledger,'h1_transfer',.30)
            expected='validated_before_training' if a.validate_only else 'sealed'
            if process['returncode'] or done['status']!=expected:
                raise RuntimeError('H1 paid path stopped without retry: '+seed)
            if fits()>22:raise ValueError('Fit ceiling exceeded')
            print(dict(outer=seed,status=done['status'],fits=done['actual_classifier_fits'],new_API=done['real_API_requests']),flush=True)
        after=ledger.project_summary()
        if after['requests']-before['requests']>6 or source!=hashes(ROOT):raise ValueError('Request/source identity changed')
        save(a.output/'results.json',dict(paths=results,project_usage_before=before,project_usage_after=after,
            known_incremental_cost_rmb=after['known_course_cost_rmb']-before['known_course_cost_rmb']))
        save(a.output/'completion.json',dict(status='validated' if a.validate_only else 'paid_paths_sealed',
            actual_classifier_fits=fits(),real_API_requests=after['requests']-before['requests'],
            A_cumulative_fits_after=1566+fits(),source_unchanged=True,outer_scores_computed=0,
            elapsed_seconds=time.monotonic()-start))
    except Exception as error:
        after=ledger.project_summary()
        save(a.output/'results.json',dict(paths=results,project_usage_before=before,project_usage_after=after))
        save(a.output/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=fits(),
            real_API_requests=after['requests']-before['requests'],A_cumulative_fits_after=1566+fits(),
            automatic_retry=False,outer_scores_computed=0,elapsed_seconds=time.monotonic()-start));raise


if __name__=='__main__':main()
