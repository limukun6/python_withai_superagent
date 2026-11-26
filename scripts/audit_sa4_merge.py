"""Audit merge provenance and export only metadata from the completed synthetic smoke."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--private-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    def git(*flags): return subprocess.run(['git',*flags],capture_output=True,check=True,text=True,encoding='utf-8').stdout.strip()
    def save(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def read(path): return json.loads(path.read_text(encoding='utf-8'))
    def tree(ref,prefix):
        return {line.split('\t',1)[1][len(prefix):]:line.split()[2]
                for line in git('ls-tree','-r',ref,'--',prefix).splitlines()}
    base=tree('b0944ae','superagent/')
    mirror=tree('c6ea0b7','releases/autonomous-integration-2026-11-26/superagent/')
    A=['semantic_contract','output_recipe','format_intake','predict','partial_classification','events']
    B=['ab_methods','experiment_policy','exploration_rule','planner_review','conditional_review','proposal_value',
       'feedback_reflection','scheduled_review','protected_stop','opening_evidence','splits','task_budget','experiment_llm']
    preserved={}
    for ref,items in [('0d09760',A),('b0944ae',B)]:
        for name in items:
            path='superagent/'+name+'.py'
            # hash-object with --path applies the same Git text filters as the commit.
            actual=git('hash-object','--path='+path,path)
            expected=git('rev-parse',ref+':'+path)
            preserved[path]=dict(source=ref,unchanged=actual==expected,git_blob=actual)
    total=0
    arms={}
    for arm in ['default','combined']:
        src=args.private_run/arm/'agent'; dst=args.output/arm; dst.mkdir()
        for name in ['classifier_fits.json','final_selection.json','prediction_preflight.json',
                     'task_opening.json','policy_review.json','stop_review.json']:
            if (src/name).is_file(): save(dst/name,read(src/name))
        fits=read(src/'classifier_fits.json'); total+=fits['attempted_classifier_fits']
        arms[arm]=dict(fits=fits['attempted_classifier_fits'],
            successful_classifier_calls=sum(c['status']=='completed' for w in fits['workers'] for c in w['calls']))
    passed=(base==mirror and all(row['unchanged'] for row in preserved.values()) and total==4
            and all(arm['successful_classifier_calls']==2 for arm in arms.values()))
    result=dict(status='passed' if passed else 'failed',B_root='b0944ae',B_release='c6ea0b7',A='0d09760',
        mirrored_modules=len(base),all_release_module_blobs_match=base==mirror,
        mirrored_entry_match=git('rev-parse','b0944ae:unified_agent.py')==git('rev-parse','c6ea0b7:releases/autonomous-integration-2026-11-26/unified_agent.py'),
        preserved_modules=preserved,actual_classifier_fits=total,arms=arms,new_audit_fits=0,real_API_requests=0,
        audit_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    save(args.output/'results.json',result); save(args.output/'completion.json',result)
    print(json.dumps(dict(status=result['status'],mirrored_modules=len(base),preserved_modules=len(preserved),physical_fits=total)))
    return not passed


if __name__=='__main__': raise SystemExit(main())
