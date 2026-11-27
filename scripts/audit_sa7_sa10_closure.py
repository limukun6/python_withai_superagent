"""Read-only final resource and delivery reconciliation for this automatic continuation window."""
import argparse
from contextlib import closing
import json
from pathlib import Path
import sqlite3

from scripts.run_sa5_semantic_panel import read,save,sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1]
    before=sha(a.ledger);requests=[]
    with closing(sqlite3.connect(a.ledger.resolve().as_uri()+'?mode=ro',uri=True)) as db:
        db.row_factory=sqlite3.Row;rows=[dict(r) for r in db.execute('SELECT * FROM requests')]
    for row in rows:
        if row['scope'] not in ('sa7_multiturn_20261126','sa8_real_delivery_20261126'):continue
        if row['status']!='known':raise ValueError('New unresolved window request')
        resolution=json.loads(row['payload'])['resolution']
        if row['course_units']!=1000*(resolution['input_tokens']+4*resolution['output_tokens']):raise ValueError('Price settlement mismatch')
        requests.append(dict(id=row['id'],scope=row['scope'],input_tokens=resolution['input_tokens'],output_tokens=resolution['output_tokens'],course_cost_rmb=row['course_units']/1e9))
    paths=[('V2','runs/sa8_live_01/V2'),('V1','runs/sa9_recovery_01/V1'),('V3','runs/sa9_recovery_01/V3')]
    fits=0;models=[]
    final=read(root/'research/sa9_recovery_01/results.json')
    for case,name in paths:
        folder=root/name/'agent';fit=read(folder/'classifier_fits.json');selection=read(folder/'final_selection.json')
        if fit['attempted_classifier_fits']!=4 or fit['completed_classifier_fits']!=4:raise ValueError('Incomplete physical accounting')
        if sum(w['attempted_classifier_fits'] for w in fit['workers'])!=4:raise ValueError('Worker fit count mismatch')
        if sha(folder/selection['model_file'])!=selection['model_sha256']:raise ValueError('Frozen model digest mismatch')
        observed=next(r for r in final['cases'] if r['case']==case)
        if observed['model_sha256']!=selection['model_sha256']:raise ValueError('Closure/model mismatch')
        fits+=4;target=a.output/case;target.mkdir()
        save(target/'classifier_fits.json',fit);save(target/'final_selection.json',selection)
        models.append(dict(case=case,physical_fits=4,family=selection['family'],model_sha256=selection['model_sha256'],
            prediction_rows=observed['prediction_rows'],development_correct=observed['development_correct'],
            development_score=observed['development_score']))
    compatibility=read(root/'research/sa10_model_compatibility/results.json')
    if compatibility['status']!='passed' or len(compatibility['cases'])!=3:raise ValueError('Current-source compatibility missing')
    result=dict(status='passed',physical_classifier_fits=fits,real_API_requests=len(requests),requests=requests,
        input_tokens=sum(r['input_tokens'] for r in requests),output_tokens=sum(r['output_tokens'] for r in requests),
        known_incremental_course_cost_rmb=sum(round(r['course_cost_rmb']*1e9) for r in requests)/1e9,
        ledger_requests=len(rows),ledger_known_course_cost_rmb=sum(r['course_units'] or 0 for r in rows)/1e9,
        ledger_unknown=sum(r['status']=='unknown' for r in rows),ledger_pending=sum(r['status']=='reserved' for r in rows),
        new_unknowns=0,models=models,compatibility_models=3,
        audit_classifier_fits=0,audit_API_requests=0,ledger_unchanged=before==sha(a.ledger),team_total_verified=False,
        scope='SA7 live cases plus SA8 original/SA9 recovery; no first-attempt 3/3 or causal accuracy-improvement claim')
    save(a.output/'results.json',result);save(a.output/'completion.json',result)
    print(json.dumps({k:result[k] for k in ['status','physical_classifier_fits','real_API_requests','known_incremental_course_cost_rmb','ledger_requests','ledger_unknown']}))
    return 0


if __name__=='__main__':raise SystemExit(main())
