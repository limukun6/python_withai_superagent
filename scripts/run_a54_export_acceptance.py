"""Preselected full-portfolio and partial packages predicted in isolated source processes."""
import argparse
from pathlib import Path
import subprocess
import sys
import time

from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.ab_run import sources
from tools.export_model_bundle import export


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('catalog-study','partial-study','output'):p.add_argument('--'+name,type=Path,required=True)
    args=p.parse_args();root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();frozen=sources();results=[]
    if read(args.catalog_study/'completion.json')['status']!='completed':raise ValueError('Completed catalogue study required')
    save(root/'protocol.json',dict(batch='A54-isolated-source-export',source_files=frozen,runner_sha256=sha(Path(__file__)),
        exporter_sha256=sha(Path(__file__).resolve().parents[1]/'tools/export_model_bundle.py'),actual_classifier_fit_limit=0,
        real_api_request_limit=0,external_labels_opened=False,fresh_dependency_install_tested=False))
    cwd=root/'isolated_workdir';cwd.mkdir()
    cases=[(name,args.catalog_study/(name+('_full' if name in ('coupon','photo') else '_shared')),
            args.catalog_study/'evaluation'/name/'features.zip',args.catalog_study/(name+('_full' if name in ('coupon','photo') else '_shared'))/'predictions/rows.csv')
            for name in ('coupon','preference','sms','photo','bean')]
    cases.append(('coupon_partial',args.partial_study/'coupon_fault',args.catalog_study/'evaluation/coupon/features.zip',args.partial_study/'coupon_fault_original.csv'))
    for name,run,features,expected in cases:
        bundle=export(run,root/name);prediction=root/name/'isolated_predictions/rows.csv'
        with (root/(name+'.log')).open('w',encoding='utf-8') as log:
            code=subprocess.run([sys.executable,'-I',str(bundle/'predict.py'),'--features',str(features),'--output',str(prediction)],
                cwd=cwd,stdout=log,stderr=subprocess.STDOUT,timeout=60).returncode
        if code:raise ValueError('Isolated-source prediction failed: '+name)
        if prediction.read_bytes()!=expected.read_bytes():raise ValueError('Package prediction differs: '+name)
        result=read(root/name/'completion.json');result.update(task=name,isolated_predictions_identical=True,
            predictions_sha256=sha(prediction),original_repository_source_not_required=True,fresh_dependency_install_tested=False)
        results.append(result);save(root/'results.json',results)
    save(root/'completion.json',dict(status='passed',packages=len(results),partial_packages=1,actual_classifier_fits=0,real_api_requests=0,
        project_source_isolated=True,fresh_dependency_install_tested=False,source_unchanged=sources()==frozen,elapsed_seconds=time.monotonic()-start))
    print(read(root/'completion.json'))
