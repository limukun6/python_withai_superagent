"""Compare archived B inputs with independently regenerated A12 training and feature packets."""
import argparse
from pathlib import Path

from scripts import run_b1_ab_acceptance as rec
from superagent.ab_run import sha
from superagent.data import load_development_task
from tools.ab_upgrade_equivalence import upgrade


def check(args):
    root = args.output.resolve()
    root.mkdir(parents=True,exist_ok=False)
    rec.offline(root)
    old = rec.read(rec.ROOT/'research/b1_06_mrpc/protocol.json')['tasks']['45101_extended']
    new = rec.read(args.canonical.parent/'protocol.json')['mrpc']
    rec.write(root/'protocol.json',dict(batch='BA12-legacy',old=old['files'],canonical=new['files'],
              plan=sha(rec.ROOT/'docs/batches/BA12_legacy_plan.md'),runner=sha(Path(__file__)),
              converter=sha(rec.ROOT/'tools/ab_upgrade_equivalence.py'),fits_allowed=0,model_api_limit=0))
    try:
        exact = {}
        for name in ('train.zip','features.zip','reversed.zip'):
            if sha(args.archived/name)!=old['files'][name] or sha(args.canonical/name)!=new['files'][name]:
                raise ValueError('Archived or regenerated packet changed')
            upgrade(args.archived/name,root/name,'0','1')
            if sha(root/name)!=new['files'][name]:
                raise ValueError('Migrated packet differs from the independent rebuild')
            exact[name]=sha(root/name)
        named = args.named/'train.zip'
        upgrade(named,root/'named.zip','different','same')
        typed = load_development_task(root/'named.zip')
        if set(typed.y_train)!={0,1}:
            raise ValueError('Archived named labels did not become equivalent/different')
        rejections = {}
        for name,file,negative,positive in [('unmapped',named,'no','yes'),('directional',args.preference/'train.zip','0','1')]:
            dest=root/(name+'.zip')
            try:
                upgrade(file,dest,negative,positive)
            except ValueError as error:
                rejections[name]=str(error)
            if name not in rejections or dest.exists():
                raise ValueError('Expected rejection before writing a converted packet')
        rec.write(root/'results.json',dict(byte_identical=exact,named_rows=len(typed.y_train),rejections=rejections))
        rec.write(root/'completion.json',dict(status='passed',real_packets_identical=len(exact),fits=0,
                  model_api_requests=0,outer_scores=0))
    except Exception as error:
        rec.write(root/'completion.json',dict(status='failed',reason=str(error),fits=0,model_api_requests=0))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('archived','canonical','named','preference','output'):
        p.add_argument('--'+key,type=Path,required=True)
    check(p.parse_args())
