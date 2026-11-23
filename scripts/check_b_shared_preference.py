"""Check the shared catalog and CLI using an archived primary fit and label-free input."""
import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.pipeline import Pipeline

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))


def unpack(workspace,output):
    sys.path.insert(0,str(workspace/'work/mainline_error_specialist'))
    from superagent.pretrained_choice import digest
    src=workspace/'outputs/mainline-release-models/task2-wide/run'
    trial=joblib.load(src/'trial_7.joblib')
    payload=joblib.load(src/'inner_data.joblib')
    assert digest(src/'inner_data.joblib')==trial['prediction_record']['evaluation_payload_sha256']
    m=trial['model']
    assert m.encoder_class.model_id=='sentence-transformers/all-MiniLM-L12-v2'
    joblib.dump(dict(word=m.word,char=m.char,scaler=m.scaler,classifier=m.classifier,rows=payload['X_search'],
        predictions=trial['search_predictions'],probabilities=trial['search_scores'],
        source_sha256=digest(src/'trial_7.joblib')),output)


def run(workspace,output):
    from scripts.run_b_local_component_pair import SHAPE, check, command, packets, read, save
    from superagent import ab_run
    from superagent.ab_methods import TEXT_DEEP,TEXT_EDIT,PREFERENCE_CATALOG_ENV
    from superagent.ab_sentence import DeepPreferenceFeatures
    from superagent.external_assets import model_resources,verify_resources
    from superagent.ab_edit_views import edit_view
    from superagent.ab_catalog import capabilities_for
    from superagent.prediction_evidence import fingerprint

    base=REPO/'runs/local_components_02'
    spec=read(base/'original/state_1.json')['task']
    schema=spec['input_schema']
    symmetric=spec|{'input_schema':schema|{'pair_semantics':'symmetric','pair_label_one':'equivalent'}}
    env={'SUPERAGENT_MINILM_WEIGHTS':str(workspace/'work/pretrained_weights/all-MiniLM-L6-v2'),
         'SUPERAGENT_DEEP_MINILM_WEIGHTS':str(workspace/'work/pretrained_weights/all-MiniLM-L12-v2'),
         'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false',
         'AB_B1_OFFLINE':'1','PYTHONPATH':str(REPO/'scripts/ab_acceptance_guard')}
    menus={}
    evidence={}
    with patch.dict(os.environ,env), tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        for mode in ('standard','deep','deep-edit'):
            with patch.dict(os.environ,{PREFERENCE_CATALOG_ENV:mode}):
                caps=capabilities_for(spec,train_rows=6608)
                menu=list(caps)
                menus[mode]=dict(size=len(menu),new=[n for n in (TEXT_DEEP,TEXT_EDIT) if n in caps])
                for n in menus[mode]['new']:
                    check(caps[n]['classifier_fits']==caps[n]['refit_classifier_fits']==1,'Wrong fit count')
                    check(caps[n]['swap_label_rule']=='flips','Wrong pair semantics')
                check(not {TEXT_DEEP,TEXT_EDIT}&capabilities_for(symmetric,train_rows=6608).keys(),'Wrong symmetric menu')
        check([menus[k]['new'] for k in menus]==[[],[TEXT_DEEP],[TEXT_DEEP,TEXT_EDIT]],'Wrong preference menu')
        with patch.dict(os.environ,{PREFERENCE_CATALOG_ENV:'deep-edit','SUPERAGENT_DEEP_MINILM_WEIGHTS':str(tmp)}):
            check(not {TEXT_DEEP,TEXT_EDIT}&capabilities_for(spec,train_rows=6608).keys(),'Unverified encoder admitted')
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--workspace',str(workspace),
            '--unpack',str(tmp/'fixture.joblib')],check=True,env=os.environ.copy(),cwd=tmp)
        fixture=joblib.load(tmp/'fixture.joblib')
        rows=fixture['rows']
        packets(rows,range(len(rows)),False,tmp/'features.zip')
        check(len(rows)==1372,'Wrong archived primary split')
        examples=[r for r in rows if edit_view(*r)[1]][:8]+[r for r in rows if not edit_view(*r)[1]][:8]
        swapped=[(b,a) for a,b in examples]
        scores=np.asarray(fixture['probabilities'])
        expected=np.asarray(fixture['predictions'])
        for name in (TEXT_DEEP,TEXT_EDIT):
            use_edit=name==TEXT_EDIT
            feature=DeepPreferenceFeatures('pair_text',edit_context=use_edit)
            feature.word_,feature.char_,feature.scaler_=fixture['word'],fixture['char'],fixture['scaler']
            feature.usage_=dict(input_tokens=0,feature_transform_calls=0,api_calls=0,api_rmb=0)
            model=copy.deepcopy(fixture['classifier'])
            if use_edit:
                model.coef_=np.pad(model.coef_,((0,0),(0,384)))
                model.n_features_in_+=384
                forward=feature.transform(examples)
                backward=feature.transform(swapped)
                check(abs(forward+backward).max()<1e-5,'Edit features are not antisymmetric')
                lengths=np.asarray(abs(forward[:,-384:]).sum(axis=1)).ravel()
                check(np.all(lengths[:8]>0) and np.all(lengths[8:]==0),'Edit activation does not match the input')
            pipeline=Pipeline([('features',feature),('classifier',model)])
            resource,pkgs=model_resources(pipeline)
            check(len(resource)==1 and resource[0]['name']=='minilm_l12','Wrong delivery dependency')
            verify_resources(resource)
            probability=pipeline.predict_proba(rows)
            p=probability if scores.ndim==2 else probability[:,1]
            np.testing.assert_allclose(p,scores,atol=3e-5,rtol=0)
            run_dir=tmp/name
            run_dir.mkdir()
            joblib.dump(dict(model=pipeline,kind='pair_text',candidate=name),run_dir/'selected_model.joblib')
            save(run_dir/'task_spec.json',dict(task_kind='pair_text_binary',model_kind='pair_text',metric='accuracy',
                 metric_spec=spec['metric'],input_schema=SHAPE))
            save(run_dir/'final_selection.json',dict(selected_experiment='frozen_weight_preflight',family=name,
                 model_file='selected_model.joblib',model_format='candidate_bundle_v1',test_accessed=False,
                 model_sha256=ab_run.sha(run_dir/'selected_model.joblib'),task_spec_sha256=ab_run.sha(run_dir/'task_spec.json')))
            status,_=command(['predict',str(tmp/'features.zip'),'--run-dir',str(run_dir),'--output',str(run_dir/'rows.csv')],
                             run_dir/'predict.log',os.environ.copy(),600)
            if status:
                raise RuntimeError((run_dir/'predict.log').read_text()[-1800:])
            with (run_dir/'rows.csv').open() as f:
                result=list(csv.DictReader(f))
            np.testing.assert_array_equal([int(v['row_id']) for v in result],np.arange(len(rows)))
            preds=[int(v['prediction']) for v in result]
            np.testing.assert_array_equal(preds,expected)
            check(not read(run_dir/'prediction_manifest.json')['test_labels_read'],'CLI opened labels')
            evidence[name]=dict(rows=len(rows),prediction_sha256=fingerprint(preds),
                max_probability_error=float(abs(p-scores).max()),required_resources=[r['name'] for r in resource],
                edit_coefficients='all zero for delivery check' if use_edit else None)
        output.parent.mkdir(parents=True,exist_ok=True)
        save(output,dict(menus=menus,frozen_classifier_sha256=fixture['source_sha256'],checks=evidence,
            mirrored_active_inactive_examples=[8,8],new_supervised_fits=0,model_api_calls=0,official_validation=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace',type=Path,required=True)
    dest=parser.add_mutually_exclusive_group(required=True)
    dest.add_argument('--unpack',type=Path)
    dest.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.unpack:
        unpack(args.workspace.resolve(),args.unpack)
    else:
        run(args.workspace.resolve(),args.output)
