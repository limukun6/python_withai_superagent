"""Develop first; retained cases require an exact source freeze and a single-use marker."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

from scripts.a_batch_common import hashes,save
from scripts.run_a10_real_task_pilot import read,sha

ROOT=Path(__file__).resolve().parents[1]


def legacy(case):
    from superagent.prose_claims import check_prose
    from superagent.parameter_semantics import check_parameter_semantics
    sentence=case['legacy_sentence'];kind=case['claim']['kind'];state=case['state'];name=case['selected_candidate'];caps=case['capabilities']
    if case['group']=='parameter_direction':
        report=check_parameter_semantics({'reason':sentence},{'id':name,'params':caps[name]['params']},state['history'],
              {'classifier':caps[name]['estimator'],'source':'shared_ab_catalogue'},state['task']['kind'])
        items=report['items'];statuses=[x['status'] for x in items]
    else:
        report=check_prose({'reason':sentence},{'classifier':caps[name]['estimator']},state['history'],state['task']['kind'])
        # Recognizing that a trial exists does not verify its outcome, causal claim,
        # task type or a complete multi-number comparison proposition.
        target='classifier' if case['group']=='object_identity' else 'percentage' if case['claim'].get('ref','').endswith('.search.accuracy') else None
        items=[i for i in report['items'] if i['kind']==target] if target else []
        statuses=[{'unverifiable':'unknown','informational':'unknown'}.get(x['status'],x['status']) for x in items]
    status='contradicted' if 'contradicted' in statuses else 'confirmed' if statuses and all(s=='confirmed' for s in statuses) else 'unknown'
    return dict(status=status,items=len(items),scope='Existing bounded atom checker, not full natural-language certification')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',type=Path,required=True)
    p.add_argument('--split',choices=['development','retained'],required=True);p.add_argument('--freeze',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();source=hashes(ROOT);prepared=read(a.cases/'completion.json');casefile=a.cases/(a.split+'.json')
    save(out/'protocol.json',dict(batch='A145-'+a.split,source_files=source,runner_sha256=sha(Path(__file__)),
         cases_sha256=sha(casefile),plan_sha256=sha(ROOT/'docs/batches/A145_grounded_protocol_execution_plan_20261124_zh.md'),
         maximum_fits=0,maximum_API_calls=0))
    try:
        if sha(casefile)!=prepared[a.split+'_sha256']:raise ValueError('Frozen case bytes differ')
        if a.split=='retained':
            if a.freeze is None or read(a.freeze)['source_files']!=source:raise ValueError('Retained evaluation requires matching frozen source')
            marker=a.cases/'retained_evaluation_started.json'
            with marker.open('x',encoding='utf-8') as stream:
                json.dump(dict(output=str(out),source_files=source,cases_sha256=sha(casefile)),stream)
        from scripts import a38_predict_guard
        from superagent.fact_registry import build_registry,check_claim
        rows=[]
        for case in read(casefile):
            registry=build_registry(case['state'],case['capabilities'],case['binding']);report=check_claim(case['claim'],registry)
            rows.append(dict(id=case['id'],group=case['group'],expected=case['expected'],actual=report['status'],
                 passed=report['status']==case['expected'],report=report,legacy=legacy(case)))
        classes={label:[r for r in rows if r['expected']==label] for label in ('confirmed','contradicted','unknown')}
        wrong=sum(r['actual']=='confirmed' for r in classes['contradicted']);unknown=sum(r['actual']=='confirmed' for r in classes['unknown'])
        valid=sum(r['actual']=='confirmed' for r in classes['confirmed']);gate=wrong==unknown==0 and valid>=11
        result=dict(rows=rows,passed=sum(r['passed'] for r in rows),total=len(rows),false_confirmations=wrong,
             unknown_certified=unknown,valid_confirmed=valid,valid_total=len(classes['confirmed']),eligible=gate,
             legacy_status_counts=dict(Counter(r['legacy']['status'] for r in rows)),
             evidence_scope='Program-oracle controlled records; structured coverage is not unrestricted language understanding')
        if source!=hashes(ROOT):raise ValueError('Production source changed during evaluation')
        save(out/'results.json',result);save(out/'completion.json',dict(status='passed' if gate else 'gate_failed',
             exact_case_agreement=result['passed'],cases=len(rows),eligible=gate,actual_classifier_fits=0,real_API_requests=0,
             source_unchanged=True,elapsed_seconds=time.monotonic()-start,protocol_sha256=sha(out/'protocol.json'),results_sha256=sha(out/'results.json')))
        print(json.dumps({k:result[k] for k in ('passed','total','false_confirmations','unknown_certified','valid_confirmed','eligible','legacy_status_counts')}))
    except Exception as e:
        save(out/'completion.json',dict(status='failed',error=str(e),actual_classifier_fits=0,real_API_requests=0));raise


if __name__=='__main__':main()
