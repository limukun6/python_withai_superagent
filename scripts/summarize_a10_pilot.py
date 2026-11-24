"""Archive compact, non-binary A10 evidence and print per-task development results."""
import argparse
import json
from pathlib import Path
import shutil

from scripts.a_batch_common import save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    protocol=read(args.run/'protocol.json')
    completion=read(args.run/'completion.json')
    results=read(args.run/'results.json')
    args.output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for result in results:
        name=result['name'];outcome=result['outcome']
        meta=next(t for t in protocol['tasks'] if t['name']==name)
        actions=result['actions']
        row=dict(task=name,primary_metric=meta['metric'],score=result.get('verification',{}).get('score'),
            held_rows=meta['task']['held_rows'],selected=outcome['selected_candidate'],
            candidate_count=len(meta['capabilities']),attempted_fits=outcome['classifier_fit_accounting']['attempted_classifier_fits'],
            completed_fits=outcome['classifier_fit_accounting']['completed_classifier_fits'],
            cli_seconds=outcome['elapsed_seconds'],train_process_seconds=result['train_process_seconds'],
            export_seconds=result.get('export_seconds'),action_counts=outcome['action_counts'],
            explored=[a['candidate_id'] for a in actions if a['action']=='explore'],
            promotions=outcome['confirmation']['b_scheduler_review']['promotions'],
            reload_identical=result.get('relocated_prediction_identical',False),api_requests=outcome['new_api_requests'])
        rows.append(row)
        dest=args.output/name;dest.mkdir()
        for file in ('run_outcome.json','actions.json','final_selection.json','export_verification.json',
                     'classifier_fits.json'):
            source=args.run/name/file
            if source.is_file():shutil.copy2(source,dest/file)
    # All actual row IDs are retained in the protocol; binary payloads/data/models are excluded.
    for file in ('protocol.json','completion.json','all_models_frozen.json','physical_fits.jsonl','environment.txt'):
        if (args.run/file).is_file():shutil.copy2(args.run/file,args.output/file)
    save(args.output/'summary.json',dict(completion=completion,tasks=rows,
        scope='Development pilot only, no fixed-vs-dynamic causal comparison'))
    print(json.dumps(dict(completion=completion,tasks=rows),indent=2))


if __name__=='__main__':main()
