"""Reexport four genuine frozen Q models and check the same grader used by the formal B matrix."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

from scripts import run_b1_ab_acceptance as rec, run_b_c1_matrix as matrix


def run(args):
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    rec.offline(out)
    os.environ.update(SUPERAGENT_MINILM_WEIGHTS=str(args.weights.resolve()),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    protocol=matrix.freeze_inputs(args.portable)
    reference=rec.read(rec.ROOT/'research/bq_01_portfolios/results.json')['paths']
    templates=[(name,arm) for name in matrix.TASKS for arm in ('original','expanded')]
    models={}
    rec.write(out/'protocol.json',dict(batch='BC1-05',matrix_protocol=rec.digest(protocol),
              code=matrix.ab_run.sha(Path(__file__)),plan=matrix.ab_run.sha(rec.ROOT/'docs/batches/BC1_05_grading_plan.md'),
              paths=templates,fits_allowed=0,model_api_limit=0))
    try:
        for name,arm in templates:
            key=name+'_'+arm
            run=args.q/key/'run'
            selection=rec.read(run/'final_selection.json')
            models[key]=matrix.ab_run.sha(run/selection['model_file'])
            if models[key]!=reference[key]['model_sha256'] or models[key]!=selection['model_sha256'] or not selection['refit']:
                raise ValueError('Q model differs from its actual frozen identity')
        rec.write(out/'all_models_frozen.json',models)
        predictions={n:{} for n in matrix.TASKS}
        hashes={}
        for name,arm in templates:
            key=name+'_'+arm
            target=out/key/'rows.csv'
            with (out/(key+'.log')).open('x') as log:
                subprocess.run([sys.executable,'-m','superagent','predict',str(args.portable/name/'delivery/features.zip'),
                                 '--run-dir',str(args.q/key/'run'),'--output',str(target)],check=True,cwd=rec.ROOT,
                                 stdout=log,stderr=subprocess.STDOUT,timeout=300)
            hashes[key]=matrix.ab_run.sha(target)
            if hashes[key]!=reference[key]['delivery']['csv_sha256']:
                raise ValueError('Fresh unordered/ordered delivery differs from archived export: '+key)
            predictions[name][arm]=matrix.read_export(target,protocol['tasks'][name]['held_rows'])
        scores={}
        for name in matrix.TASKS:
            scores[name]=matrix.grade_task(name,args.portable/name,protocol['tasks'][name],
                                           getattr(args,name),predictions[name])
            for arm,observed in scores[name].items():
                original=reference[name+'_'+arm]['delivery']
                if any(original[k]!=observed[k] for k in ('rows','primary_metric','accuracy','score')):
                    raise ValueError('Measured grading differs on '+name+'/'+arm)
        staged=out/'empty_matrix'
        staged.mkdir()
        rec.write(staged/'protocol.json',protocol)
        common=['--matrix',str(staged),'--portable',str(args.portable),'--weights',str(args.weights)]
        controls={'score':(['--headlines',str(out/'missing_headlines'),'--bean',str(out/'missing_bean'),'--output',str(out/'unused')],'Scoring requires every frozen model'),
                  'run':(['--headlines-id','qualification-headlines','--bean-id','qualification-bean','--ledger',str(out/'missing.sqlite'),
                          '--api-key-file',str(out/'missing_key')],'Budget ledger is not initialized')}
        clean_env=dict(os.environ)
        clean_env.pop('SUPERAGENT_OFFLINE',None)
        for mode,(flags,message) in controls.items():
            complete=subprocess.run([sys.executable,'-m','scripts.run_b_c1_matrix',mode,*common,*flags],cwd=rec.ROOT,
                        env=clean_env,capture_output=True,text=True,timeout=120)
            if not complete.returncode or message not in complete.stderr:
                raise ValueError('Formal command did not stop at the expected boundary: '+mode)
        if (staged/'arms').exists() or (staged/'budget_context.json').exists() or (out/'unused').exists():
            raise ValueError('A blocked command started an arm or opened a scoring directory')
        rec.write(out/'results.json',dict(models=models,prediction_sha256=hashes,scores=scores,
                  command_rejections={m:v[1] for m,v in controls.items()}))
        rec.write(out/'completion.json',dict(status='passed',actual_frozen_predictions=4,
                  historical_scores_reproduced=4,formal_c1_paths_executed=0,fits=0,model_api_requests=0))
    except Exception as error:
        rec.write(out/'completion.json',dict(status='failed',reason=str(error),fits=0,model_api_requests=0))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('portable','weights','q','headlines','bean','output'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    for key in ('portable','weights','q','headlines','bean'):
        setattr(a,key,getattr(a,key).resolve())
    run(a)
