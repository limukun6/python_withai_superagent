"""Run only the already-qualified replication with the pre-registered export-only repair."""
import argparse
from pathlib import Path

from scripts import compare_joint_feedback as joint
from scripts.study_feedback_design import ARMS, frozen
from scripts.unified_released_study import ROOT, read, require, save


def run(args):
    study=args.root/'development'
    approval=read(ROOT/'research/feedback_design_1127/export_repair.json')
    registration=read(study/'registered.json')['frozen']
    require(registration==approval['initial'] and frozen()==approval['fixed'], 'Unregistered source change')
    require(args.case in approval['eligible']==read(study/'replications.json')['run'], 'Not qualified by internal evidence')
    require(not (study/'scored.json').exists(), 'Outer results were already disclosed')
    joint.run(args,expected_fingerprint=registration,additional_sealed_directory=args.previous)
    save(study/args.case/args.arm/'export_fix_binding.json',dict(runtime=frozen()['agent'],original=registration['agent'],
        commit=approval['fixed_commit'],scope=approval['scope']))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=lambda s:Path(s).resolve(),required=True)
    p.add_argument('--previous',type=lambda s:Path(s).resolve(),default=ROOT.parent.parent/'outputs/joint-feedback-20261127')
    p.add_argument('--weights',type=lambda s:Path(s).resolve(),default=ROOT.parent/'pretrained_weights')
    p.add_argument('--case',required=True)
    p.add_argument('--arm',required=True,choices=ARMS)
    run(p.parse_args())
