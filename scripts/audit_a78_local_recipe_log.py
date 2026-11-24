"""Read-only recovery of A77 event-kind statistics; preserves original results and source identities."""
import argparse
import json
from pathlib import Path
import time
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha

ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);began=time.monotonic();ref=a.reference.resolve();run=ref/'run'
    protected={str(f.relative_to(ref)):sha(f) for f in ref.rglob('*') if f.is_file()}
    save(out/'protocol.json',dict(batch='A78',reference=str(ref),original_A77_protocol=read(ref/'protocol.json'),
        original_completion=read(ref/'completion.json'),runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(ROOT/'docs/batches/A78_local_recipe_log_reaudit_plan.md'),actual_classifier_fits=0,real_api_requests=0))
    events=[json.loads(line) for line in (run/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    history={r['trial']:r for r in read(run/'history.json')};caps=read(run/'run_manifest.json')['capabilities'];results=[]
    for event in events:
        if event.get('kind')!='local_text_recipe_proposal':continue
        trial=event['trial'];proof=event['proposal'];row=history[trial]
        admitted=[e for e in events if e.get('kind')=='scheduler_action_admitted' and e['action']['candidate_id']==proof['chosen_id']
                  and e['state_sha256']==__import__('superagent.scheduler_bridge',fromlist=['fingerprint']).fingerprint(read(run/f'state_{trial}.json'))]
        if not admitted or row['candidate']!=proof['chosen_id'] or row['status']!='completed':raise ValueError('Proposed action was not actually admitted and completed')
        if proof['chosen_spec']!=caps[proof['chosen_id']]['params']:raise ValueError('Recipe executable parameters differ')
        if any(i>=trial or i not in history or history[i]['status']!='completed' for i in proof['evidence_trials']):raise ValueError('Recipe cites future or failed evidence')
        if not set(proof['evidence_trials'])<=set(admitted[0]['action']['evidence_trials']):raise ValueError('Dispatcher action dropped recipe evidence')
        results.append(dict(trial=trial,candidate=proof['chosen_id'],evidence_trials=proof['evidence_trials'],admitted=True,completed=True,proposal=proof))
    if not results:raise ValueError('No actual local recipe event')
    if any(sha(ref/name)!=digest for name,digest in protected.items()):raise ValueError('Original output changed')
    save(out/'results.json',dict(local_recipe_proposals=len(results),records=results,original_resource_accounting_unchanged=True,
        original_classifier_fits=read(ref/'completion.json')['actual_classifier_fits'],CSV_delivery_passed=read(ref/'results.json')['CSV_byte_identical']))
    save(out/'completion.json',dict(status='passed',actual_classifier_fits=0,real_api_requests=0,A_cumulative_fits_after=1239,
        original_A77_outputs_unchanged=True,local_recipe_proposals=len(results),elapsed_seconds=time.monotonic()-began))
