"""Resample all three registered domain comparisons using fixed paired predictions."""
import argparse
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from scripts import run_b1_ab_acceptance as record
from scripts.audit_b_kernel_grouped_effect import grouped
from scripts.run_b_c1_matrix import read_export
from scripts.run_b_kernel_domain_transfer import read_original
from superagent import ab_run
from superagent.splits import sample_groups
from tools.ab_registered_rows import content_digest


def audit(previous, archive, output):
    output.mkdir(parents=True,exist_ok=False)
    record.offline(output)
    published = record.ROOT/'research/b_text_kernel_domains'
    for name in ('preflight','results','protocol','completion','all_six_models_frozen'):
        if ab_run.sha(previous/(name+'.json'))!=ab_run.sha(published/(name+'.json')):
            raise ValueError('Frozen parent experiment differs from the published record')
    record.write(output/'protocol.json',dict(batch='B-TEXT-KERNEL-06',source_results_sha256=ab_run.sha(published/'results.json'),
        driver_sha256=ab_run.sha(Path(__file__)),bootstrap_code_sha256=ab_run.sha(record.ROOT/'scripts/audit_b_kernel_grouped_effect.py'),
        plan_sha256=ab_run.sha(record.ROOT/'docs/batches/B_TEXT_KERNEL_06_domain_uncertainty_plan.md'),
        seed=112204,resamples=4000,new_fit_limit=0,model_api_limit=0))
    try:
        domains = read_original(archive)
        pairs = record.read(published/'results.json')['pairs']
        preflight = record.read(published/'preflight.json')
        results = {}
        for domain,rows in domains.items():
            x,y = zip(*rows)
            truth = np.asarray(y)
            if content_digest(x,truth)!=preflight[domain]['held_content_sha256']:
                raise ValueError('Public domain identities changed')
            refs = pairs[domain]['grades']
            groups = sample_groups(SimpleNamespace(kind='single_text',X_train=x))
            predictions = []
            for arm in ('original','extended'):
                case = previous/(domain+'_'+arm)
                exports = [case/f'export_{i}/rows.csv' for i in range(2)]
                if any(ab_run.sha(p)!=refs[arm]['csv_sha256'] for p in exports):
                    raise ValueError('Native or relocated fixed predictions changed')
                predictions.append(np.asarray(read_export(exports[0],len(rows)),dtype=truth.dtype))
            results[domain] = grouped(truth,predictions,groups,'macro_f1',refs)
        record.write(output/'results.json',results)
        record.write(output/'completion.json',dict(status='completed',comparisons=len(results),resamples_per_pair=4000,
                    new_fits=0,new_predictions=0,new_model_api_requests=0,new_course_rmb=0))
    except Exception as error:
        record.write(output/'completion.json',dict(status='failed',reason=str(error)))
        raise


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('previous','archive','output'):
        parser.add_argument('--'+name,required=True,type=Path)
    args = parser.parse_args()
    audit(args.previous.resolve(),args.archive.resolve(),args.output.resolve())
