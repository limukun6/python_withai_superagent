"""Apply the frozen engineering criteria to four historical paired local tasks and withhold formal selection."""
import argparse
from pathlib import Path

from scripts import run_b1_ab_acceptance as rec
from scripts.measure_b1_headroom import archive_for
from scripts.replay_b1_observed_actions import SEEDS,TASKS
from superagent.ab_run import sha
from tools.ab_deployment_criteria import assess, ranking_key


def same_evidence(left,right):
    fields=('source','semantics','row_ids_sha256','splits','holdout_sha256','catalog_sha256','candidate_ids')
    if any(left[k]!=right[k] for k in fields):
        raise ValueError('Method catalogue, task or registered examples differ')


def run(args):
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    rec.offline(out)
    rec.write(out/'protocol.json',dict(batch='BC1-06',tasks=list(TASKS),seeds=list(SEEDS),paths=16,
              code=sha(Path(__file__)),rules=sha(rec.ROOT/'tools/ab_deployment_criteria.py'),
              plan=sha(rec.ROOT/'docs/batches/BC1_06_selection_plan.md'),fits_allowed=0,model_api_limit=0))
    try:
        pairs,sources=[],set()
        for name in TASKS:
            archived=archive_for(name)
            folder=getattr(args,name)
            protocol=rec.read(archived/'protocol.json')
            sources.add(protocol['source_sha256'])
            if sha(folder/'protocol.json')!=sha(archived/'protocol.json') or protocol['api_requests_allowed']!=0:
                raise ValueError('Expected an unchanged matched local batch')
            records={r['name']:r for r in rec.read(archived/'results.json')['paths']}
            for split in SEEDS:
                fixed,dynamic=[f'{split}_{p}' for p in ('fixed','adaptive')]
                same_evidence(protocol['tasks'][fixed],protocol['tasks'][dynamic])
                for key in (fixed,dynamic):
                    item=records[key]
                    run=folder/key/'run'
                    outcome=rec.read(run/'run_outcome.json')
                    if not (outcome['requested_planner_completed'] and outcome['new_api_requests']==0 and
                            outcome['elapsed_seconds']==item['seconds'] and
                            sha(run/'selected_model.joblib')==item['model_sha256'] and
                            sha(folder/key/'pred_0/rows.csv')==item['delivery']['csv_sha256']):
                        raise ValueError('Actual run time, export or model differs from archived receipt')
                a,b=[records[k] for k in (fixed,dynamic)]
                if (a['delivery']['primary_metric']!=b['delivery']['primary_metric'] or
                        a['delivery']['original_row_ids_sha256']!=b['delivery']['original_row_ids_sha256']):
                    raise ValueError('Compared outcomes belong to different metrics or evaluation examples')
                pairs.append(dict(task=name,split=split,metric=a['delivery']['primary_metric'],
                             baseline_score=a['delivery']['score'],score=b['delivery']['score'],
                             baseline_seconds=a['seconds'],seconds=b['seconds'],requests=0))
        if len(sources)!=1:
            raise ValueError('Cannot pool different source releases')
        q=rec.read(rec.ROOT/'research/bq_01_portfolios/protocol.json')['tasks']
        try:
            same_evidence(q['headlines_original'],q['headlines_expanded'])
        except ValueError as error:
            negative=str(error)
        else:
            raise ValueError('Different method catalogues were admitted as an allocation comparison')
        result=assess(pairs)
        rec.write(out/'results.json',dict(source=next(iter(sources)),pairs=pairs,historical_diagnostic=result,
                  ranking_values=dict(zip(('negative_mean_pp','negative_worst_pp','actual_requests','machine_seconds'),ranking_key(result),strict=True)),
                  menu_change_rejected=negative,official_deployment_policy=None,
                  reason='Only the four earlier local development blocks are present; the final joint matrix is unavailable'))
        rec.write(out/'completion.json',dict(status='measured',matched_local_paths=16,matched_pairs=len(pairs),
                  final_policy_selected=False,fits=0,new_predictions=0,model_api_requests=0))
    except Exception as error:
        rec.write(out/'completion.json',dict(status='failed',reason=str(error),fits=0,model_api_requests=0))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in (*TASKS,'output'):
        p.add_argument('--'+key,type=Path,required=True)
    run(p.parse_args())
