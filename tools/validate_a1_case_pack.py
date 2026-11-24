"""Validate an A1 contribution before evaluating a checker; never generates expected labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

CATEGORIES={'task_semantics','configuration','trial_identity','metric_partition','negation_future','slice_scope','memory_state','failure_comparison'}
OPERATIONS={'semantics','configuration','measurement','prose','slice','memory','overall'}


def validate(rows,role):
    errors=[]
    if not isinstance(rows,list):return ['Expected a JSON array of independently labeled cases']
    if len(rows)!=48:errors.append('Exactly 48 contributed cases are required')
    ids=[];counts=Counter();inputs=set();labels={g:set() for g in CATEGORIES}
    for i,row in enumerate(rows):
        if not isinstance(row,dict):errors.append(f'Case {i} must be an object');continue
        identifier=row.get('id');group=row.get('category');expected=row.get('expected')
        if not isinstance(identifier,str) or not identifier:errors.append(f'Case {i} needs an ID')
        elif role=='B' and not identifier.startswith('B_'):errors.append(f'Case {identifier}: independent pack ID must start B_')
        ids.append(identifier)
        if group not in CATEGORIES:errors.append(f'Case {i} has an unknown category');continue
        counts[group]+=1;labels[group].add(expected)
        if expected not in ('confirmed','contradicted','unknown'):errors.append(f'Case {identifier}: unsupported expected status')
        if row.get('operation') not in OPERATIONS or not isinstance(row.get('payload'),dict):errors.append(f'Case {identifier}: invalid operation/payload')
        if row.get('author')!=role:errors.append(f'Case {identifier}: author does not match contribution role')
        wanted='A_development' if role=='A' else 'B_independent_reserved'
        if row.get('subset')!=wanted:errors.append(f'Case {identifier}: expected subset {wanted}')
        if not isinstance(row.get('expectation_basis'),str) or len(row['expectation_basis'].strip())<16:
            errors.append(f'Case {identifier}: needs a substantive predeclared label basis')
        signature=json.dumps([group,row.get('operation'),row.get('payload')],sort_keys=True,ensure_ascii=False)
        if signature in inputs:errors.append(f'Case {identifier}: duplicate input within category')
        inputs.add(signature)
    if len({str(v) for v in ids})!=len(ids):errors.append('Duplicate case IDs')
    for group in sorted(CATEGORIES):
        if counts[group]!=6:errors.append(f'{group}: expected 6 cases, observed {counts[group]}')
        if labels[group]!={'confirmed','contradicted','unknown'}:errors.append(f'{group}: include valid, conflicting and insufficient-evidence cases')
    return errors


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cases',type=Path,required=True);p.add_argument('--role',choices=['A','B'],required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=json.loads(a.cases.read_text(encoding='utf-8-sig'));errors=validate(rows,a.role)
    result=dict(status='invalid' if errors else 'valid',role=a.role,cases=len(rows) if isinstance(rows,list) else None,
        cases_sha256=hashlib.sha256(a.cases.read_bytes()).hexdigest(),errors=errors,actual_classifier_fits=0,real_api_requests=0,
        authorship_and_blindness_independently_verified=False,checker_evaluated=False)
    with a.output.open('x',encoding='utf-8') as file:json.dump(result,file,indent=2,ensure_ascii=False)
    print(json.dumps(result));raise SystemExit(bool(errors))
