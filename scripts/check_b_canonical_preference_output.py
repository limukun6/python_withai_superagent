"""Relocate two extra inference-only exports and verify the native CLI and probabilities."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import joblib
import numpy as np

from scripts import run_b1_ab_acceptance as record, run_b1_05_pair_pilot as pilot
from scripts.run_b_a19_released_delivery import write_packet
from scripts.run_b_c1_matrix import read_export
from superagent import ab_run
from superagent.data import load_prediction_features
from superagent.models import probability_with_model
from tools.ab_export_canonical_preference import export


def probe(args):
    selection=record.read(args.model/'final_selection.json')
    model_path=args.model/selection['model_file']
    if ab_run.sha(model_path)!=selection['model_sha256'] or selection['output_adapter']['new_fits']!=0:
        raise ValueError('Reloaded artifact differs from its provenance')
    spec=record.read(args.model/'task_spec.json')
    bundle=joblib.load(model_path)
    pairs=load_prediction_features(args.features,'pair_text',spec)
    model=bundle['model']
    classes,p=probability_with_model('pair_text',model,pairs)
    reverse,back=probability_with_model('pair_text',model,[(b,a) for a,b in pairs])
    if not np.array_equal(classes,reverse) or not np.isfinite(p).all() or not np.isfinite(back).all():
        raise ValueError('Inconsistent classes or nonfinite probabilities')
    pos=list(classes).index(1)
    error=np.abs(p[:,pos]+back[:,pos]-1.)
    selected=np.random.default_rng(112208).choice(len(pairs),size=97,replace=False)
    parts=[selected[:1],selected[1:48],selected[48:96],selected[96:]]
    chunks=np.concatenate([model.predict_proba([pairs[i] for i in part]) for part in parts])
    delta=float(np.max(np.abs(chunks-p[selected])))
    diagonal=[(pairs[i][0],pairs[i][0]) for i in selected[:5]]
    diagonal_p=model.predict_proba(diagonal)
    count=int(np.sum(np.argmax(p,axis=1)!=np.asarray(read_export(args.labels,len(pairs)),dtype=int)))
    if float(error.max())>1e-12 or delta>.00002 or not np.all(diagonal_p==.5) or count:
        raise ValueError('Adapted probabilities, source labels, diagonal or batching failed')
    record.write(args.output,dict(rows=len(pairs),model_sha256=selection['model_sha256'],
        positive_probability_sha256=record.sha(p[:,pos].astype('<f8').tobytes()),max_swap_probability_error=float(error.max()),
        max_batch_probability_difference=delta,identical_pair_probabilities=diagonal_p.tolist(),probability_csv_label_mismatches=count))


def run(args):
    out,origin,packets=args.output.resolve(),args.previous.resolve(),args.packets.resolve()
    out.mkdir(parents=True,exist_ok=False)
    record.offline(out)
    os.environ.update(SUPERAGENT_OFFLINE='1',SUPERAGENT_MINILM_WEIGHTS=str(args.weights.resolve()),
        OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    os.environ.pop('SUPERAGENT_MINILM_CACHE_DIR',None)
    registered=record.ROOT/'research/b_text_kernel_titles'
    for key in ('all_models_frozen','results','preflight','protocol'):
        if ab_run.sha(origin/(key+'.json'))!=ab_run.sha(registered/(key+'.json')):
            raise ValueError('Frozen primary study differs from published receipts')
    if ab_run.sha(packets/'preflight.json')!='f41552bfd5d4780cd4a64db1024dc6e81a8c67beb11130fd7a4eaff7fdbb3d95':
        raise ValueError('Registered title queries differ')
    record.write(out/'protocol.json',dict(batch='B-CANONICAL-OUTPUT-01',parent_results_sha256=ab_run.sha(registered/'results.json'),
        plan_sha256=ab_run.sha(record.ROOT/'docs/batches/B_CANONICAL_OUTPUT_01_plan.md'),driver_sha256=ab_run.sha(Path(__file__)),
        output_code_sha256=ab_run.sha(record.ROOT/'superagent/ab_preference_output.py'),
        export_code_sha256=ab_run.sha(record.ROOT/'tools/ab_export_canonical_preference.py'),new_fit_limit=0,model_api_limit=0))
    try:
        jobs={}
        for key in ('45101','45102'):
            batch=out/key
            batch.mkdir()
            src=origin/key/'relocated'
            initial=record.read(src/'final_selection.json')
            expected=record.read(registered/'all_models_frozen.json')[key]['extended']
            features=packets/key/'features.zip'
            pre=record.read(packets/'preflight.json')['entries'][key+'_original']
            if ab_run.sha(features)!=pre['features_sha256'] or initial['model_sha256']!=expected:
                raise ValueError('Expected unlabeled query or frozen source model changed')
            selection=export(src,batch/'adapted')
            moved=batch/'moved'
            shutil.copytree(batch/'adapted',moved)
            data=load_prediction_features(features,'pair_text',record.read(src/'task_spec.json'))
            reverse=batch/'reversed.zip'
            write_packet(reverse,'headlines',[(b,a) for a,b in data])
            query=[('native',batch/'adapted',features),('relocated',moved,features),('reverse',moved,reverse)]
            for name,folder,input_path in query:
                pilot.child(['-m','superagent','predict',input_path,'--run-dir',folder,'--output',batch/name/'rows.csv'],batch/(name+'.log'),300)
            native=batch/'native/rows.csv'
            raw=origin/key/'export_0/rows.csv'
            if (ab_run.sha(native)!=ab_run.sha(batch/'relocated/rows.csv') or ab_run.sha(native)!=ab_run.sha(raw)
                    or ab_run.sha(raw)!=record.read(registered/'results.json')['pairs'][key]['grades']['extended']['csv_sha256']):
                raise ValueError('Native direction or relocation changed hard-label exports')
            fwd=np.asarray(read_export(native,len(data)),dtype=int)
            rev=np.asarray(read_export(batch/'reverse/rows.csv',len(data)),dtype=int)
            distinct=np.asarray([a!=b for a,b in data])
            if np.any(fwd[distinct]==rev[distinct]):
                raise ValueError('At least one pair of distinct titles failed to flip')
            for name in ('adapted','moved'):
                pilot.child(['-m','scripts.check_b_canonical_preference_output','probe','--model',batch/name,'--features',features,
                            '--labels',native,'--output',batch/(name+'_probabilities.json')],batch/(name+'_probabilities.log'),300)
            check=record.read(batch/'adapted_probabilities.json')
            if check!=record.read(batch/'moved_probabilities.json') or ab_run.sha(src/initial['model_file'])!=expected:
                raise ValueError('Relocation changed output probabilities or original artifact changed')
            jobs[key]=dict(candidate=initial['family'],original_model_sha256=expected,export_sha256=ab_run.sha(native),
                source_label_changes=0,distinct_pairs=int(distinct.sum()),reverse_label_mismatches=0,source_selection_score_scope=selection['output_adapter']['selection_score_scope'],**check)
            record.write(out/'checkpoint.json',jobs)
        if record.trace(out):
            raise ValueError('Frozen packaging started training')
        record.write(out/'results.json',jobs)
        record.write(out/'completion.json',dict(status='completed',paths=len(jobs),fresh_cli_exports=6,fresh_probability_processes=4,
                    new_training=0,new_model_api_requests=0,new_course_rmb=0,outer_labels_read=False))
    except Exception as error:
        record.write(out/'completion.json',dict(status='failed',reason=str(error)))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_subparsers(dest='mode',required=True)
    p=modes.add_parser('study')
    for option in ('previous','packets','weights','output'):
        p.add_argument('--'+option,type=Path,required=True)
    p=modes.add_parser('probe')
    for option in ('model','features','labels','output'):
        p.add_argument('--'+option,type=Path,required=True)
    args=parser.parse_args()
    if args.mode=='study':
        run(args)
    else:
        probe(args)
