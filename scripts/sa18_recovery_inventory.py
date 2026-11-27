"""Read-only inventory of current task-ending branches, no classifier fits or model API."""
import argparse
import ast
import json
from pathlib import Path

from scripts.run_sa5_semantic_panel import save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--ledger',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    names=['superagent/semantic_contract.py','superagent/autonomous_run.py','superagent/partial_classification.py','superagent/task_questions.py']
    identities={name:sha(root/name) for name in names}
    save(a.output/'protocol.json',dict(batch='SA18-recovery-readonly-inventory',source_sha256=identities,
        plan_sha256=sha(root/'docs/batches/SA18_RECOVERY_REALIGNMENT_20261127_zh.md'),
        runner_sha256=sha(Path(__file__)),maximum_classifier_fits=0,maximum_real_API_requests=0))
    evidence=[]
    for name in names:
        source=(root/name).read_text(encoding='utf-8');tree=ast.parse(source)
        for function in [n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))]:
            if function.name=='semantic_training':
                for item in ast.walk(function):
                    if isinstance(item,ast.While):
                        evidence.append(dict(file=name,function=function.name,line=item.lineno,finding='bounded_semantic_loop',code=ast.unparse(item.test)))
                    if isinstance(item,ast.Raise) and item.exc is not None and 'Semantic contract probe did not pass' in ast.unparse(item.exc):
                        evidence.append(dict(file=name,function=function.name,line=item.lineno,finding='intake_exhaustion_raises',code=ast.unparse(item)))
            if function.name=='_train':
                for item in ast.walk(function):
                    if isinstance(item,ast.Return) and isinstance(item.value,ast.Call) and isinstance(item.value.func,ast.Name) and item.value.func.id=='record_input_failure':
                        evidence.append(dict(file=name,function=function.name,line=item.lineno,finding='input_failure_returns_from_training',code=ast.unparse(item)))
    ledger=None
    if a.ledger:
        from superagent.task_budget import TaskBudgetLedger
        ledger=TaskBudgetLedger(a.ledger).project_summary()
    result=dict(status='inventory_complete_plan_only',source_unchanged=all(sha(root/name)==value for name,value in identities.items()),
        runtime_recovery_controller_implemented=False,evidence=evidence,ledger_summary=ledger,
        actual_classifier_fits=0,real_API_requests=0,course_cost_rmb=0,
        conclusion='Current per-proposal feedback is not yet a cross-stage autonomous recovery controller. Verify recovery and correct delivery, not rejection counts.')
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
