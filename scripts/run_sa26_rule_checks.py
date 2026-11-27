"""SA26 no-fit, no-network regression of integrated A/B entry and delivery."""
import argparse
import hashlib
import json
from pathlib import Path
import socket
import time
import unittest
from unittest.mock import patch


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];a.output.mkdir(parents=True,exist_ok=False)
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    def save(name,value):(a.output/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    # Reuse SA24's fixed module selection and explicit exclusions without executing its runner.
    import ast
    tree=ast.parse((root/'scripts/run_sa24_image_checks.py').read_text(encoding='utf-8'))
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    values={}
    for node in function.body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('modules','excluded'):values[target.id]=ast.literal_eval(node.value)
    modules=values['modules']+['tests.test_task_declarations','tests.test_confusion_specialist','tests.test_reliable_feedback','tests.test_external_assets','tests.test_joint_delivery','tests.test_rule_recovery']
    excluded=values['excluded']|{'tests.test_task_declarations.TaskDeclarations.test_formal_solve_obeys_the_rule_and_reloads'}
    paths=list((root/'superagent').glob('*.py'))+[root/'unified_agent.py',Path(__file__),root/'tests/test_joint_delivery.py',root/'tests/test_rule_recovery.py']
    identity={p.relative_to(root).as_posix():sha(p) for p in paths}
    save('protocol.json',dict(batch='SA26-joint-regression',base='91f1a8c',source_sha256=identity,
        modules=modules,excluded_fit_fixtures=sorted(excluded),maximum_classifier_fits=0,maximum_campus_API_requests=0,
        plan_sha256=sha(root/'docs/batches/SA26_RULE_RECOVERY_PLAN_20261127_zh.md')))
    def flatten(suite):
        for test in suite:
            if isinstance(test,unittest.TestSuite):yield from flatten(test)
            else:yield test
    selected=[];removed=[]
    for test in flatten(unittest.TestLoader().loadTestsFromNames(modules)):
        if test.id() in excluded:removed.append(test.id())
        else:selected.append(test)
    if set(removed)!=excluded:raise ValueError('Declared exclusions changed')
    from superagent.ab_run import Executor
    from superagent.task_constraints import permitted
    worker=Executor.worker
    def no_fit_worker(self,payload,name,destination,timeout,**kwargs):
        if not permitted(self.caps[name],self.restrictions):
            return worker(self,payload,name,destination,timeout,**kwargs)
        raise AssertionError('SA26 no-fit batch')
    clock=time.monotonic()
    with (a.output/'tests.log').open('w',encoding='utf-8') as stream,patch.object(Executor,'worker',no_fit_worker),patch.object(socket,'create_connection',side_effect=AssertionError('SA26 no-network batch')):
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.TestSuite(selected))
    unchanged=identity=={n:sha(root/n) for n in identity}
    record=dict(status='passed' if result.wasSuccessful() and unchanged else 'failed',tests_run=result.testsRun,
        failures=len(result.failures),errors=len(result.errors),excluded_fit_fixtures=removed,source_unchanged=unchanged,
        actual_classifier_fits=0,campus_API_requests=0,course_cost_rmb=0,elapsed_seconds=time.monotonic()-clock)
    save('results.json',record);save('completion.json',record);print(json.dumps(record));return not result.wasSuccessful()


if __name__=='__main__':raise SystemExit(main())
