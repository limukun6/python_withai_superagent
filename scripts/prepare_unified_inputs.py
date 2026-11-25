"""Reproduce the three exact train-only packets using solely the two released archives."""
import argparse
from pathlib import Path
import zipfile

from scripts.unified_released_study import EVIDENCE, archive, make_zip, read, require, sha, text_packet, title_rows


def prepare(packages, output):
    expected=read(EVIDENCE/'protocol.json')
    table=archive(packages,'table')
    titles=archive(packages,'titles')
    require(sha(table)==expected['archives']['table'] and sha(titles)==expected['archives']['titles'],'Released archive changed')
    output.mkdir(parents=True,exist_ok=False)
    with zipfile.ZipFile(table) as z:
        members=('C_train.npy','y_train.npy','Readme.md')
        make_zip(output/'table.zip',[(n,z.read('in_vehicle_coupon_recommendation/'+n)) for n in members])
    raw=title_rows(titles,'train',labels=True)
    reserve=set(read(EVIDENCE/'title_reserve_ids.json'))
    text_packet(output/'titles.zip',[row for i,row in enumerate(raw) if i not in reserve],True)
    text_packet(output/'titles_full.zip',raw,True)
    for tag, record in expected['training'].items():
        require(sha(output/(tag+'.zip'))==record['packet'],'Training packet differs: '+tag)
    print('Reproduced all three sealed training packets.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--packages',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    prepare(a.packages,a.output)
