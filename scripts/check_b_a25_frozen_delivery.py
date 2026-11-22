"""Check A25 on four frozen Released classifiers, a symmetric development model, and H1 training schemas."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

from scripts import run_b1_ab_acceptance as rec
from scripts.run_b_c1_matrix import read_export
from superagent import ab_run
from superagent.ab_sentence import verified_directory
from superagent.contracts import TaskSpec
from superagent.data import load_development_task, load_prediction_features


def predict(run, packet, destination, model_hash, expected_hash, rows):
    selection=rec.read(run/'final_selection.json')
    if selection['model_sha256']!=model_hash or ab_run.sha(run/selection['model_file'])!=model_hash:
        raise ValueError('Frozen classifier differs')
    csv=destination/'rows.csv'
    destination.mkdir()
    with (destination/'process.log').open('x') as log:
        process=subprocess.run([sys.executable,'-m','superagent','predict',str(packet),'--run-dir',str(run),
                  '--output',str(csv)],stdout=log,stderr=subprocess.STDOUT,cwd=rec.ROOT,timeout=300)
    if process.returncode:
        tail=(destination/'process.log').read_text().strip().splitlines()[-1]
        return dict(status='prediction_failed',exit_code=process.returncode,exception=tail)
    read_export(csv,rows)
    digest=ab_run.sha(csv)
    return dict(status='identical' if digest==expected_hash else 'changed',csv_sha256=digest,rows=rows)


def run(args):
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    rec.offline(output)
    os.environ.update(SUPERAGENT_OFFLINE='1',SUPERAGENT_MINILM_WEIGHTS=str(args.weights.resolve()),
                      OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
    published=rec.ROOT/'research/ba19_released_delivery'
    source=rec.digest(ab_run.sources())
    rec.write(output/'protocol.json',dict(batch='BA25-01',production_sha256=source,
        references_sha256=ab_run.sha(published/'all_models_frozen.json'),
        driver_sha256=ab_run.sha(Path(__file__)),plan_sha256=ab_run.sha(rec.ROOT/'docs/batches/BA25_01_joint_delivery_plan.md'),
        fit_cap=0,model_api_cap=0,new_score_cap=0))
    try:
        if verified_directory() is None:
            raise ValueError('Pinned encoder unavailable')
        inputs=rec.read(published/'preflight.json')
        frozen=rec.read(published/'all_models_frozen.json')
        outcomes=rec.read(published/'results.json')['paths']
        released={}
        for key,model in frozen.items():
            task=key.split('_')[0]
            packet=args.released/task/'features.zip'
            if ab_run.sha(packet)!=inputs[task]['features_sha256']:
                raise ValueError('Released holdout features differ')
            released[key]=predict(args.released/key/'run',packet,output/key,model,outcomes[key]['csv_sha256'],inputs[task]['held_rows'])
        current=load_development_task(args.staged/'preference/train_only.zip')
        first=load_prediction_features(args.query,current.kind,current.spec)
        if current.spec.get('input_format')!='csv' or list(first)!=current.X_train[:31]:
            raise ValueError('The fresh A25 legacy task contract failed its unlabelled 31-row round trip')
        old_eq=rec.ROOT/'research/ba12_equivalence'
        ref=rec.read(old_eq/'protocol.json')
        h1={}
        for seed,old in ref['h1']['partitions'].items():
            packet=args.equivalence/('training_'+seed+'.zip')
            if ab_run.sha(packet)!=old['training_packet_sha256']:
                raise ValueError('PAWS training-only packet differs')
            fit=load_development_task(packet)
            contract=TaskSpec.from_task(fit).record()
            full=ab_run.registered_capabilities(contract,len(fit.y_train),'full')
            base=ab_run.registered_capabilities(contract,len(fit.y_train),'baseline')
            with patch.dict(os.environ,{'SUPERAGENT_MINILM_WEIGHTS':''}):
                cpu=ab_run.registered_capabilities(contract,len(fit.y_train),'full')
            if (fit.X_val is not None or fit.y_val is not None or len(fit.y_train)!=old['training']
                    or list(full)!=old['methods'] or rec.digest(full)!=old['catalog_sha256']
                    or base or list(cpu)!=[k for k in full if k!='unordered_minilm_lexical']
                    or any(c['swap_label_rule']!='preserves' or c['classifier_fits']!=1 or c['refit_classifier_fits']!=1
                           for c in full.values())):
                raise ValueError('PAWS catalogue eligibility, swap rule or fit cost changed')
            h1[seed]=dict(optimization_rows=len(fit.y_train),native_full=list(full),without_weights=list(cpu),
                          baseline_count=len(base),swap_label_rule='preserves',fit_per_trial=1,final_refit_cost=1)
        mrpc=ref['mrpc']
        receipt=rec.read(old_eq/'results.json')['mrpc']['delivery']
        symmetric={}
        for name in ('features.zip','reversed.zip'):
            packet=args.equivalence/'mrpc'/name
            if ab_run.sha(packet)!=mrpc['files'][name]:
                raise ValueError('MRPC blind development input differs')
            symmetric[name]=predict(args.equivalence/'mrpc/run',packet,output/('mrpc_'+packet.stem),
                receipt['model_sha256'],receipt['csv_sha256'],receipt['rows'])
        if rec.trace(output) or rec.digest(ab_run.sources())!=source:
            raise ValueError('Audit changed the production sources or trained a model')
        rec.write(output/'results.json',dict(released= released,fresh_legacy_schema=dict(rows=31,ordered_inputs_equal=True,
            input_format=current.spec['input_format']),H1_training_schema=h1,MRPC_frozen=symmetric))
        failures=[k for k,v in {**released,**symmetric}.items() if v['status']!='identical']
        rec.write(output/'completion.json',dict(status='compatibility_gap' if failures else 'passed',failed_exports=failures,
            independent_export_processes=6,H_fits=0,H_scores=0,H_predictions=0,
            fresh_classifier_fits=0,model_api_requests=0,new_scores=0))
    except Exception as error:
        rec.write(output/'completion.json',dict(status='failed',error=str(error),fresh_classifier_fits=len(rec.trace(output)),model_api_requests=0))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ('released','staged','query','equivalence','weights','output'):
        parser.add_argument('--'+flag,type=Path,required=True)
    run(parser.parse_args())
