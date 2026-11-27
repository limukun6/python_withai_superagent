"""Serve the original immutable fitted model under the verified legacy selection binding, without training again."""
import argparse
from pathlib import Path
import os

from scripts import compare_joint_feedback as joint
from scripts.unified_released_study import ROOT, env, read, require, sha
from scripts.verify_anonymous_planner import execute


def restore(root,previous,case):
    study=root/'development'
    arm='feedback-design'
    folder=study/case/arm
    source=folder/'delivery/predictions.csv'
    first=read(source.parent/'prediction_manifest.json')
    run=folder/'run'
    agent=run/'agent'
    selection=read(agent/'final_selection.json')
    outcome=read(agent/'run_outcome.json')
    require(first['delivery_phase']=='recovered_first' and 'does not match its frozen task and selection' in first['delivery_issue']
            and selection['family']=='class_pair_refinement' and sha(agent/selection['model_file'])==selection['model_sha256'],
            'Recovery applies only to the archived mismatched family label')
    receipt=read(study/'registered.json')['cases'][case]
    query=folder.parent/'query.zip'
    require(sha(query)==receipt['query_sha256'],'Changed prediction input')
    result=folder/'export-recovery'
    result.mkdir()
    settings=env(ROOT.parent/'pretrained_weights',folder/'physical.jsonl')
    for key in joint.CONFIG:settings.pop(key,None)
    settings.update(PYTHONPATH=os.pathsep.join(map(str,(joint.GUARD.parent,ROOT))),
                    JOINT_OUTER_ANSWERS=str(study/'evaluation'),JOINT_UPSTREAM_SOURCES=str(root/'sources'),
                    JOINT_ADDITIONAL_SEALED_DIRECTORY=str(previous),JOINT_PREDICTION_INPUT=str(query),
                    REFLECTION_STUDY_LIVE='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    export=result/'delivery/predictions.csv'
    execute(['predict',query,'--run-dir',run,'--output',export],result/'export.log',settings)
    require(sha(source)==first['prediction_sha256'],'Original partial output changed')
    info=dict(original_phase=first['delivery_phase'],original_sha256=sha(source),original_issue=first['delivery_issue'],
              verified_original_weights=selection['model_sha256'],new_fits=0,new_requests=0,
              seconds_basis='original training controller time, excluding separate export repair')
    joint.seal_run(folder,run,export,query,settings,receipt,case,arm,outcome['wall_seconds'],result,recovery=info)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=lambda s:Path(s).resolve())
    p.add_argument('case')
    p.add_argument('--previous',type=lambda s:Path(s).resolve(),default=ROOT.parent.parent/'outputs/joint-feedback-20261127')
    args=p.parse_args()
    restore(args.root,args.previous,args.case)
