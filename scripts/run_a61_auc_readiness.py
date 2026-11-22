"""No-fit AUC readiness: artificial counterexamples and real fitted-model probability order."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scripts import a38_predict_guard  # Disable supported classifier fit and all network calls.
from scripts.a_batch_common import save
from scripts.run_a10_real_task_pilot import read,sha
from superagent.metrics import MetricSpec
from superagent.models import predict_with_model,probability_with_model
from superagent.prediction_evidence import fingerprint


def main(args):
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=False);start=time.monotonic();rows=[]
    source=Path(__file__).resolve().parents[1]
    save(root/'protocol.json',dict(batch='A61',runner_sha256=sha(Path(__file__)),
        plan_sha256=sha(source/'docs/batches/A61_auc_readiness_plan.md'),reference_protocol_sha256=sha(args.reference/'protocol.json'),
        classifier_fitting_forbidden=True,network_forbidden=True,outer_validation_accessed=False,formal_test_or_H_opened=False))
    try:
        metric=MetricSpec('roc_auc',(0,1));y=[0,0,1,1];hard=[0,0,0,0]
        good=np.array([[.9,.1],[.8,.2],[.7,.3],[.6,.4]])
        bad=good[::-1]
        left=metric.evaluate(y,hard,good,[0,1])['score'];right=metric.evaluate(y,hard,bad,[0,1])['score']
        if left!=1 or right!=0 or not np.array_equal(good.argmax(axis=1),bad.argmax(axis=1)):
            raise ValueError('Same-hard-label AUC counterexample failed')
        domain=MetricSpec('roc_auc',('positive','negative'))
        renamed=['positive','positive','negative','negative']
        if domain.evaluate(renamed,['positive']*4,good,['positive','negative'])['score']!=1:
            raise ValueError('Explicit positive class order not respected')
        multi=MetricSpec('roc_auc',(0,1,2));ym=[0,1,2]*2;hm=ym
        matrix=np.tile(np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8]]),(2,1))
        if multi.evaluate(ym,hm,matrix[:,[2,0,1]],[2,0,1])['score']!=1:
            raise ValueError('Multiclass probability column reordering failed')
        rejected=[]
        for name,labels,pred,scores,classes in (
            ('nan',y,hard,np.full((4,2),np.nan),[0,1]),
            ('missing_column',y,hard,good[:,:1],[0]),
            ('unnormalized',y,hard,good/2,[0,1]),
            ('negative_probability',y,hard,np.array([[1.1,-.1]]*4),[0,1]),
            ('missing_eval_class',[0]*4,hard,good,[0,1]),
            ('duplicate_classes',y,hard,good,[0,0])):
            try:metric.evaluate(labels,pred,scores,classes)
            except ValueError:rejected.append(name)
            else:raise ValueError('Malformed AUC input accepted: '+name)
        save(root/'artificial_results.json',dict(same_hard_predictions=True,good_ranking_auc=left,bad_ranking_auc=right,
            positive_class_order_passed=True,multiclass_order_passed=True,rejected=rejected,actual_classifier_fits=0))
        reference=read(args.reference/'protocol.json')
        for name in ('coupon','preference','sms','photo','bean'):
            entry=reference['entries'][name];scope='shared' if entry['deduplicated_same_policy'] else 'full'
            run=args.reference/(name+'_'+scope);manifest=read(run/'run_manifest.json')
            history=read(run/'history.json');row=next(r for r in history if r['split_id']=='primary')
            path=run/f"trial_{row['trial']}.joblib";model_sha=sha(path);payload_path=run/'primary.joblib';digest=sha(payload_path)
            payload=joblib.load(payload_path);artifact=joblib.load(path);model=artifact['model']
            old=MetricSpec.from_payload(payload)
            if old.check_identity(artifact['metrics']['metric_spec'])['status']!='confirmed':raise ValueError('Metric identity mismatch')
            record=artifact['prediction_record'];pred=predict_with_model(payload['kind'],model,payload['X_search'])
            if (record['evaluation_payload_sha256']!=digest or record['label_sha256']!=fingerprint(payload['y_search'])
                or record['prediction_sha256']!=fingerprint(pred.tolist())):raise ValueError('Original prediction identity differs')
            classes,scores=probability_with_model(payload['kind'],model,payload['X_search'])
            auc=MetricSpec('roc_auc',old.labels)
            measured=auc.evaluate(payload['y_search'],pred,scores,classes)
            reverse=auc.evaluate(payload['y_search'],pred,scores[:,::-1],classes[::-1])
            if abs(measured['score']-reverse['score'])>1e-12:raise ValueError('Aligned class reordering changes AUC')
            if sha(path)!=model_sha or sha(payload_path)!=digest:raise ValueError('Read-only model/payload changed')
            rows.append(dict(task=name,kind=payload['kind'],candidate=row['candidate'],original_target=old.name,
                metric_scope='Posthoc train-internal probability diagnostic; not AUC-optimized search',rows=len(pred),
                diagnostic_auc=measured['score'],original_hard_accuracy=measured['accuracy'],
                class_order=classes.tolist(),aligned_column_reversal_identical=True,original_predictions_reproduced=True,
                model_artifact_sha256=model_sha,evaluation_payload_sha256=digest,
                scores_sha256=hashlib.sha256(np.ascontiguousarray(scores,dtype=np.float64).tobytes()).hexdigest(),
                manifest_sha256=sha(run/'run_manifest.json')))
            save(root/'real_model_diagnostics.json',rows)
        save(root/'completion.json',dict(status='completed',real_models=len(rows),actual_classifier_fits=0,real_api_requests=0,
            elapsed_seconds=time.monotonic()-start,formal_test_or_H_opened=False,
            AB_AUC_end_to_end_enabled=False,scope='Probability/metric readiness only'))
    except Exception as error:
        save(root/'completion.json',dict(status='stopped',error=str(error),models_completed=len(rows),
            actual_classifier_fits=0,real_api_requests=0,automatic_retry=False));raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    main(p.parse_args())
