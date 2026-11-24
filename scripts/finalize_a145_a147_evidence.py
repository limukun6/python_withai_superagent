"""Seal offline follow-up evidence and record the decision not to expand this variant."""
import argparse
import json
from pathlib import Path

from scripts.a_batch_common import hashes,save
from scripts.run_a10_real_task_pilot import read,sha

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('followup','paid','evaluation','output','api-key-file'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=hashes(ROOT)
    secret=a.api_key_file.read_text(encoding='utf-8').strip().encode();files={}
    completion=read(a.followup/'completion.json');result=read(a.followup/'results.json');evaluated=read(a.evaluation/'results.json')
    if completion['status']!='completed' or read(a.followup/'protocol.json')['source_files']!=source:
        raise ValueError('Follow-up completion/source mismatch')
    for file in a.followup.rglob('*'):
        if file.is_file() and file.suffix in ('.json','.jsonl','.log'):
            raw=file.read_bytes()
            if secret and secret in raw:raise ValueError('Credential detected; content not printed')
            relative=file.relative_to(a.followup);target=out/'followup'/relative
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw);files[relative.as_posix()]=sha(target)
    # Post-seal contextual review; original automatic flags are retained unchanged.
    legacy=read(a.paid/'flowers/natural_legacy/answer.json')
    spans=[('reason','more aggressive logistic/SVM settings'),('hypothesis','linear logistic regression misses')]
    if legacy['configuration_claim']!={'classifier':'extra_trees'} or any(text not in legacy[field] for field,text in spans):
        raise ValueError('Review no longer matches sealed answer')
    review=dict(analyst='A-side Codex; not an independent human gold annotation',review_timing='After answers were sealed',
        original_automatic_prose_flags=2,original_flags_retained=True,
        findings=[dict(field=field,quote=text,interpretation='Mentions an alternative or historical classifier, not a claim that the chosen estimator is that classifier') for field,text in spans],
        conclusion='Two likely false positives from the old narrow classifier phrase checker; do not report 2-to-0 as two real reasoning errors removed',
        controlled_comparison_limit='The injected prior proposal was changed without recomputing the old post-execution prose audit. New mode derives bound issues; this does not compare two re-audited complete paths.')
    save(out/'contextual_review.json',review)
    gate=dict(stage='A148',decision='do_not_launch_current_full_loop_variant',decision_made_after_A147=True,
        numerical_cutoff_preregistered=False,
        rationale='No stable task-quality advantage: SMS controlled diagnostic lower on both splits; flower coverage changes sign. New protocol increases cost and latency. Do not spend 44 fits/12 calls merely to expand this pilot.',
        same_task_split_evidence_not_new_task_generalization=True,
        A145_reliability_gate_passed=True,A146_real_typed_corrections=3,A146_real_semantic_repair_reasks=0,
        comparisons=result['comparisons'],by_mode=evaluated['by_mode'],
        default_protocol_changed=False,final_project_frozen=False,additional_fits_launched=0,additional_API_calls_launched=0,
        next_design='Separate knowledge-only correction from execution/measurement-required repair, so correcting a definition need not consume a low-value training action; new protocol and experiment needed before running.')
    save(out/'A148_gate_decision.json',gate)
    save(out/'archive_receipt.json',dict(status='archived',source_files=source,files_sha256=files,
        source_completion_sha256=sha(a.followup/'completion.json'),actual_classifier_fits=0,real_API_requests=0,
        followup_fits_referenced=completion['actual_classifier_fits'],ledger_or_key_copied=False,
        current_ledger_owner_reference='../a145_a146_grounded_pilot/paid/current_ledger_owner.json'))
    print(json.dumps(dict(status='archived',followup_fits=completion['actual_classifier_fits'],A148='not_launched')))


if __name__=='__main__':main()
