"""Count uniquely attributable local B worker fit receipts across experiment checkouts."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, data):
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')


def audit(work, out):
    out.mkdir(parents=True,exist_ok=False)
    save(out/'protocol.json',dict(batch='B-SHARED-TRAINING-ACCOUNTING',fit_limit=0,api_limit=0,
        source='ab_*/runs; original admission plus actual worker calls; copies of the same original event only counted once',
        driver_sha256=sha(Path(__file__).read_bytes()),
        plan_sha256=sha((Path(__file__).resolve().parents[1]/'docs/batches/B_SHARED_TRAINING_ACCOUNTING_plan.md').read_bytes())))
    originals, paths, ignored = {},set(),Counter()
    groups = {}
    try:
        for root in sorted(work.glob('ab_*/runs')):
            for parent, directories, names in os.walk(root):
                directories[:] = [d for d in directories if d not in ('.git','repo','research','tests','H1','H2','registered')]
                folder = Path(parent)
                for name in sorted(names):
                    if not name.endswith('.admission.json'):
                        continue
                    expected_name = name.removesuffix('.admission.json')+'.fits.json'
                    ledger = folder/expected_name
                    if not ledger.is_file():
                        ignored['no_worker_receipt'] += 1
                        continue
                    admission = json.loads((folder/name).read_bytes())
                    original = Path(admission['ledger_path'])
                    if not original.is_absolute() or original.name!=expected_name or not original.is_relative_to(work):
                        ignored['unattributed_origin'] += 1
                        continue
                    relative = original.relative_to(work)
                    if len(relative.parts)<4 or not relative.parts[0].startswith('ab_') or relative.parts[1]!='runs':
                        ignored['outside_B_research_scope'] += 1
                        continue
                    receipt = json.loads(ledger.read_bytes())
                    attempted,completed = receipt['attempted_classifier_fits'],receipt['completed_classifier_fits']
                    if (type(attempted) is not int or attempted!=len(receipt['calls']) or type(completed) is not int
                            or completed!=sum(c.get('status')=='completed' for c in receipt['calls'])
                            or receipt['maximum_classifier_fits']!=admission['maximum_classifier_fits']
                            or not 0<=completed<=attempted<=admission['maximum_classifier_fits']):
                        raise ValueError('Worker fit count does not reconcile')
                    if not attempted:
                        ignored['zero_attempts'] += 1
                        continue
                    location = sha(relative.as_posix().encode())
                    if location in originals:
                        if originals[location]!=receipt:
                            raise ValueError('Incompatible copies claim the same original attempt')
                        ignored['duplicate_receipt'] += 1
                        continue
                    originals[location] = receipt
                    paths.add(relative.parent)
                    group = groups.setdefault(relative.parts[0],Counter())
                    group.update(attempted=attempted,completed=completed,distinct_workers=1)
        if not originals:
            raise ValueError('No independently attributable local worker fit records')
        totals = Counter()
        for group in groups.values():
            totals.update(group)
        proof = [[k,sha(json.dumps(v,sort_keys=True,separators=(',',':')).encode())] for k,v in sorted(originals.items())]
        save(out/'results.json',dict(totals=dict(totals),distinct_run_directories=len(paths),by_original_checkout=groups,
            excluded=dict(ignored),proof_count=len(proof),identity_digest=sha(json.dumps(proof,separators=(',',':')).encode())))
        save(out/'completion.json',dict(status='completed',verified_lower_bound=totals['attempted'],new_fits=0,
            new_model_api_requests=0,new_course_rmb=0,official_test_or_H_accessed=False))
    except Exception as error:
        save(out/'completion.json',dict(status='failed',reason=str(error)))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args = parser.parse_args()
    audit(args.workspace.resolve(),args.output.resolve())
