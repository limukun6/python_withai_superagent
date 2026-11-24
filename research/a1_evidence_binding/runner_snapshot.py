"""A1-02 offline probes and regression, with immutable per-run evidence."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['baseline','fixed','regression'],required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        (out/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    save('protocol.json',dict(phase=args.phase,utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((REPO/'superagent').glob('*.py'))},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),api_requests=0,
        scope='Known-defect regression probes, not heldout task performance'))
    start=time.monotonic();rows=[]
    def record(case,category,observed,expected):
        rows.append(dict(case=case,category=category,observed=observed,expected=expected,passed=observed==expected))
        save('results.json',rows)
    try:
        if args.phase=='regression':
            suites=['tests.test_prose_claims','tests.test_estimator_facts','tests.test_measurement_claims',
                'tests.test_explanation_audit','tests.test_targeted_feedback','tests.test_architecture_contracts',
                'tests.test_hypothesis_agent','tests.test_task_semantics','tests.test_general_extensions','tests.test_confirmation']
            with (out/'tests.log').open('w',encoding='utf-8') as stream:
                result=subprocess.run([sys.executable,'-m','unittest',*suites,'-q'],cwd=REPO,
                    env={**os.environ,'SUPERAGENT_OFFLINE':'1','PYTHONIOENCODING':'utf-8'},
                    stdout=stream,stderr=subprocess.STDOUT,timeout=600)
            frozen=json.loads((out/'protocol.json').read_text(encoding='utf-8'))['source_sha256']
            current={str(p.relative_to(REPO)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((REPO/'superagent').glob('*.py'))}
            unchanged=frozen==current
            code=result.returncode if unchanged else 3
            save('completion.json',dict(status='complete' if code==0 else 'failed',
                returncode=code,source_unchanged=unchanged,elapsed_seconds=time.monotonic()-start,api_requests=0,
                experimental_classifier_fits=0,fixture_classifier_fits='not independently counted'))
            return code
        from sklearn.linear_model import RidgeClassifier,LogisticRegression
        from sklearn.svm import SVC,LinearSVR
        from superagent.estimator_facts import observed_classifier
        from superagent.prose_claims import check_prose
        for name,model,expected in [('ridge',RidgeClassifier(),'ridge'),('logistic',LogisticRegression(),'logistic'),
            ('poly',SVC(kernel='poly'),'poly_svm'),('rbf',SVC(kernel='rbf'),'rbf_svm'),
            ('linear',SVC(kernel='linear'),'linear_svm'),('sigmoid',SVC(kernel='sigmoid'),'sigmoid_svm'),
            ('regressor',LinearSVR(),None)]:
            record(name,'estimator',observed_classifier(model)['family'],expected)
        history=[dict(trial=1,status='completed',primary_metric='accuracy',search_accuracy=.6,train_accuracy=.9,search_score=.6,train_score=.9),
                 dict(trial=2,status='completed',primary_metric='macro_f1',search_accuracy=.7,train_accuracy=.95,search_score=.65,train_score=.8),
                 dict(trial=4,status='failed',search_accuracy=.99)]
        cases=[('wrong_trial','Trial 1 search accuracy was 95%.','percentage','contradicted'),
            ('correct_trial','Trial 1 search accuracy was 60%.','percentage','confirmed'),
            ('wrong_split','Trial 2 search accuracy was 95%.','percentage','contradicted'),
            ('correct_split','Trial 2 train accuracy was 95%.','percentage','confirmed'),
            ('wrong_metric','Trial 2 search macro_f1 was 70%.','percentage','contradicted'),
            ('correct_metric','Trial 2 search macro_f1 was 65%.','percentage','confirmed'),
            ('unbound_number','search accuracy 70%','percentage','unverifiable'),
            ('missing_trial','Trial 3 achieved 95%.','trial_reference','contradicted'),
            ('trial_zero','Trial 0 achieved 90%.','trial_reference','contradicted'),
            ('failed_trial','Trial 4 search accuracy was 99%.','percentage','unverifiable'),
            ('negation','The selected classifier is not logistic regression.','classifier','unverifiable'),
            ('future','Trial 3 will reach 95%.','percentage','unverifiable')]
        for name,text,kind,expected in cases:
            report=check_prose({'reason':text},{'classifier':'logistic'},history)
            item=next(r for r in report['items'] if r['kind']==kind)
            record(name,'prose',item['status'],expected)
        summary=dict(status='complete',phase=args.phase,cases=len(rows),passed=sum(r['passed'] for r in rows),
            elapsed_seconds=time.monotonic()-start,api_requests=0,classifier_fits=0)
        save('completion.json',summary);print(json.dumps(summary))
        return int(args.phase=='fixed' and not all(r['passed'] for r in rows))
    except Exception as error:
        save('completion.json',dict(status='failed',error=f'{type(error).__name__}: {error}',
            elapsed_seconds=time.monotonic()-start,api_requests=0))
        raise

if __name__=='__main__':raise SystemExit(main())
