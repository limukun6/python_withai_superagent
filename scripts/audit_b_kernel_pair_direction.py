"""Decompose swap asymmetry of real preference models and test one-pass canonical prediction."""
import argparse
import os
from pathlib import Path

import joblib
import numpy as np
from scipy.special import expit

from scripts import run_b1_ab_acceptance as record
from scripts.run_b_c1_matrix import read_export
from superagent import ab_run
from superagent.data import load_development_task, load_prediction_features


TOLERANCE = .00002


def canonical_probabilities(model, pairs):
    lower = np.asarray([a<b for a,b in pairs])
    equal = np.asarray([a==b for a,b in pairs])
    ordered = [(a,b) if a<=b else (b,a) for a,b in pairs]
    classes = model.classes_.tolist()
    p = model.predict_proba(ordered)[:,classes.index(1)]
    return np.where(equal,.5,np.where(lower,p,1.-p))


def diagnostic(model, pairs, reference):
    features,head = model.named_steps['features'],model.named_steps['classifier']
    if head.classes_.tolist()!=[0,1] or head.coef_.shape[0]!=1 or len(head.intercept_)!=1:
        raise ValueError('Expected the fitted binary preference pipeline')
    flipped=[(b,a) for a,b in pairs]
    forward,back=[features.transform(x) for x in (pairs,flipped)]
    weight=head.coef_[0]
    intercept=float(head.intercept_[0])
    score,reverse_score=[head.decision_function(x) for x in (forward,back)]
    left,right=expit(score),expit(reverse_score)
    if not np.array_equal((left>.5).astype(int),reference):
        raise ValueError('Scores do not reproduce frozen hard labels')
    columns=[('word',len(features.word_.vocabulary_)),('char',len(features.char_.vocabulary_)),
             ('style',len(features.scaler_.scale_)),('embedding',384)]
    if hasattr(features,'kernel_'):
        columns.append(('random_fourier',features.kernel_.n_components))
    if sum(n for _,n in columns)!=len(weight):
        raise ValueError('Unknown fitted feature blocks')
    combined=forward+back
    total=np.full(len(pairs),2*intercept)
    parts={}
    position=0
    for name,n in columns:
        block=combined[:,position:position+n]
        values=np.asarray(block@weight[position:position+n]).reshape(-1)
        total+=values
        parts[name]=dict(width=n,max_feature_residual=float(np.max(np.abs(block.data))) if block.nnz else 0.,
            max_absolute_logit_contribution=float(np.max(np.abs(values))),mean_logit_contribution=float(values.mean()))
        position+=n
    decomposition_error=float(np.max(np.abs(total-(score+reverse_score))))
    if decomposition_error>1.e-9:
        raise ValueError('Feature contributions do not reproduce actual score sums')
    raw_error=np.abs(left+right-1.)
    canonical=canonical_probabilities(model,pairs)
    reversed_canonical=canonical_probabilities(model,flipped)
    can_error=np.abs(canonical+reversed_canonical-1)
    ties=(np.abs(canonical-.5)<=TOLERANCE)|(np.abs(reversed_canonical-.5)<=TOLERANCE)
    diffs=np.abs(canonical-left)
    # The diagonal has equal utilities by definition, regardless of an estimated intercept.
    diagonal=[(s,s) for s in [pairs[0][0],pairs[len(pairs)//2][1],pairs[-1][0]]]
    equality=canonical_probabilities(model,diagonal)
    return dict(rows=len(pairs),intercept=intercept,twice_intercept=2*intercept,feature_blocks=parts,
        decomposition_max_error=decomposition_error,raw_swap_violations=int(np.sum(raw_error>TOLERANCE)),
        raw_max_swap_probability_error=float(raw_error.max()),raw_mean_swap_probability_error=float(raw_error.mean()),
        underlying_score_sum_range=[float(np.min(score+reverse_score)),float(np.max(score+reverse_score))],
        canonical_max_swap_probability_error=float(can_error.max()),canonical_swap_violations=int(np.sum(can_error>TOLERANCE)),
        canonical_nontie_label_swap_failures=int(np.sum(((canonical>.5)==(reversed_canonical>.5))&~ties)),
        tolerance_ties=int(ties.sum()),identical_input_half_probabilities=equality.tolist(),
        canonical_vs_original_changed_labels=int(np.sum((canonical>.5).astype(int)!=reference)),
        canonical_vs_original_mean_probability_difference=float(diffs.mean()),canonical_vs_original_max_probability_difference=float(diffs.max()))


def run(args):
    out,origin,packets=args.output.resolve(),args.previous.resolve(),args.packets.resolve()
    out.mkdir(parents=True,exist_ok=False)
    record.offline(out)
    os.environ.update(SUPERAGENT_OFFLINE='1',SUPERAGENT_MINILM_WEIGHTS=str(args.weights.resolve()))
    os.environ.pop('SUPERAGENT_MINILM_CACHE_DIR',None)
    published=record.ROOT/'research/b_text_kernel_titles'
    for n in ('protocol','preflight','all_models_frozen','results','completion'):
        if ab_run.sha(origin/(n+'.json'))!=ab_run.sha(published/(n+'.json')):
            raise ValueError('Original title study does not match the repository')
    if ab_run.sha(packets/'preflight.json')!='f41552bfd5d4780cd4a64db1024dc6e81a8c67beb11130fd7a4eaff7fdbb3d95':
        raise ValueError('Held feature registration differs')
    record.write(out/'protocol.json',dict(batch='B-TEXT-KERNEL-08',plan_sha256=ab_run.sha(record.ROOT/'docs/batches/B_TEXT_KERNEL_08_swap_diagnosis_plan.md'),
        driver_sha256=ab_run.sha(Path(__file__)),parent_results_sha256=ab_run.sha(published/'results.json'),
        seeds=[45101,45102],tolerance=TOLERANCE,fit_limit=0,model_api_limit=0))
    try:
        prior=record.read(published/'results.json')
        result={}
        for key in ('45101','45102'):
            case=origin/key
            choice=record.read(case/'relocated/final_selection.json')
            model_file=case/'relocated'/choice['model_file']
            frozen=record.read(published/'all_models_frozen.json')[key]['extended']
            query=packets/key/'features.zip'
            dev=load_development_task(case/'train.zip')
            if ab_run.sha(query)!=record.read(packets/'preflight.json')['entries'][key+'_original']['features_sha256']:
                raise ValueError('Frozen query changed')
            data=load_prediction_features(query,dev.kind,dev.spec)
            export=case/'export_0/rows.csv'
            if ab_run.sha(model_file)!=frozen or ab_run.sha(export)!=prior['pairs'][key]['grades']['extended']['csv_sha256']:
                raise ValueError('Prior winning model or native output changed')
            labels=np.asarray(read_export(export,len(data)),dtype=int)
            result[key]=dict(candidate=choice['family'],model_sha256=frozen,**diagnostic(joblib.load(model_file)['model'],data,labels))
            record.write(out/'checkpoint.json',result)
        if record.trace(out):
            raise ValueError('Unexpected fit')
        record.write(out/'results.json',result)
        record.write(out/'completion.json',dict(status='completed',frozen_paths=len(result),new_fits=0,
                    new_model_api_requests=0,new_course_rmb=0,outer_labels_read=False))
    except Exception as error:
        record.write(out/'completion.json',dict(status='failed',reason=str(error)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('previous','packets','weights','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    run(parser.parse_args())
