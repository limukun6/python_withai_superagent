"""Unlabelled routing/export fixture using unchanged A127 fitted components; no new fits."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

import joblib
import numpy as np

from scripts.a_batch_common import hashes, save
from scripts.run_a10_real_task_pilot import read, sha
from scripts.run_a122_current_agent_acceptance import environment, child

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    env=environment(out);os.environ.update(env)
    from scripts import a38_predict_guard  # fit and network are forbidden in this process too
    from scripts.run_b1_05_pair_pilot import pair_package
    from superagent.ab_run import sources
    from superagent.ab_complement import PreferenceProbabilityVote
    from superagent.ab_encoder_routing import ConfidenceRoutedPreferenceVote
    from superagent import ab_complement
    original=BASE/'runs/a127_joint_run/run'
    components=[original/'trial_1.joblib',original/'trial_3.joblib']
    source=hashes(ROOT)
    save(out/'protocol.json',dict(batch='A134',source_files=source,runner_sha256=sha(Path(__file__)),
         plan_sha256=sha(ROOT/'docs/batches/A133_A135_next_integration_and_task1_plan_20261123_zh.md'),
         original_component_sha256={str(f):sha(f) for f in components},thresholds=[0.,1.,.2],
         probability_atol=1e-6,fit_cap=0,model_api_cap=0,manual_engineering_fixture=True,
         scope='Synthetic existing-component delivery compatibility, not autonomous selection or accuracy evidence'))
    try:
        models=[joblib.load(f)['model'] for f in components]
        ordinary=[('a quiet library','zebra crossing'),('breaking world news','weather update'),
                  ('same sentence','same sentence'),('alpha wins','beta loses')]
        rows=ordinary+[(b,a) for a,b in ordinary]
        full=PreferenceProbabilityVote('pair_text',models,np.array([0,1])); expected=full.predict_proba(rows)
        results=[]; original_probability=ab_complement.probability_with_model
        for threshold in (0.,1.,.2):
            calls=[]
            def spy(kind,model,inputs):
                calls.append(dict(component=next(i for i,m in enumerate(models) if m is model),rows=len(inputs)))
                return original_probability(kind,model,inputs)
            routed=ConfidenceRoutedPreferenceVote('pair_text',models,np.array([0,1]),0,threshold)
            with patch.object(ab_complement,'probability_with_model',side_effect=spy): values=routed.predict_proba(rows)
            if routed.predict_proba([]).shape!=(0,2): raise ValueError('Empty input shape changed')
            np.testing.assert_allclose(values[:4],values[4:,::-1],atol=1e-6,rtol=0)
            np.testing.assert_allclose(values[[2,6],1],[.5,.5],atol=0,rtol=0)
            second_rows=sum(c['rows'] for c in calls if c['component']==1)
            if threshold==0 and second_rows!=0: raise ValueError('All-skip branch evaluated second model')
            if threshold==1:
                if second_rows!=len(rows): raise ValueError('Full branch skipped rows')
                np.testing.assert_allclose(values,expected,atol=1e-6,rtol=0)
                if not np.array_equal(routed.predict(rows),full.predict(rows)): raise ValueError('Full-vote labels changed')
            results.append(dict(threshold=threshold,first_rows=len(rows),second_rows=second_rows,
                                actual_calls=calls,rows_skipped=len(rows)-second_rows))
        fixture=out/'fixture';fixture.mkdir();shutil.copy2(original/'task_spec.json',fixture/'task_spec.json')
        routed=ConfidenceRoutedPreferenceVote('pair_text',models,np.array([0,1]),0,.2)
        joblib.dump(dict(kind='pair_text',candidate='confidence_routed_vote',model=routed),fixture/'selected_model.joblib')
        save(fixture/'run_manifest.json',dict(source_files=sources(),manual_engineering_fixture=True,
            original_component_sha256={str(f):sha(f) for f in components},new_classifier_fits=0,
            origin='A127 component reuse; this source only assembles the routing wrapper'))
        save(fixture/'final_selection.json',dict(selected_experiment='controlled_fixture',family='confidence_routed_vote',
             model_file='selected_model.joblib',model_format='candidate_bundle_v1',test_accessed=False,
             refit=False,delivery_status='engineering_fixture',model_sha256=sha(fixture/'selected_model.joblib'),
             task_spec_sha256=sha(fixture/'task_spec.json')))
        query=out/'query.zip';pair_package(query,rows,None,'preference','test')
        child([sys.executable,'-m','scripts.a38_predict_guard',str(query),'--run-dir',str(fixture),
               '--output',str(out/'native.csv')],out/'native.log',env)
        moved=out/'relocated';shutil.copytree(fixture,moved)
        child([sys.executable,'-m','scripts.a38_predict_guard',str(query),'--run-dir',str(moved),
               '--output',str(out/'relocated.csv')],out/'relocated.log',env)
        child([sys.executable,'-m','tools.export_model_bundle','--run-dir',str(fixture),'--output',str(out/'portable')],out/'bundle.log',env)
        child([sys.executable,'-I',str(out/'portable/bundle/predict.py'),'--features',str(query),
               '--output',str(out/'isolated.csv')],out/'isolated.log',env)
        if len({(out/f).read_bytes() for f in ('native.csv','relocated.csv','isolated.csv')})!=1:
            raise ValueError('Native, relocated and portable predictions differ')
        if (out/'physical_fits.jsonl').exists() or hashes(ROOT)!=source: raise ValueError('Unexpected fits or source drift')
        save(out/'results.json',dict(routing_checks=results,rows=len(rows),portable_predictions_identical=True,
             prediction_sha256=sha(out/'native.csv'),external_resources=read(out/'portable/bundle/bundle_manifest.json')['external_resources'],
             accuracy_scored=False,manual_engineering_fixture=True))
        save(out/'completion.json',dict(status='passed',actual_classifier_fits=0,real_API_requests=0,
             elapsed_seconds=time.monotonic()-started,source_unchanged=True,
             protocol_sha256=sha(out/'protocol.json'),results_sha256=sha(out/'results.json')))
    except Exception as e:
        save(out/'completion.json',dict(status='failed',error=str(e),actual_classifier_fits=0,real_API_requests=0,
             elapsed_seconds=time.monotonic()-started));raise
    print(json.dumps(dict(status='passed',actual_classifier_fits=0,real_API_requests=0)))


if __name__=='__main__': main()
