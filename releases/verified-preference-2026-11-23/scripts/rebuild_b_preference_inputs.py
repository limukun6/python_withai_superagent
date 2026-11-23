"""Rebuild the verified training and development inputs from the course's released training file."""
import argparse
import csv
import io
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_b_local_component_pair import ARCHIVE_SHA, MEMBER, check, packets, read, save
from superagent.ab_run import sha
from superagent.splits import grouped_partition, sample_groups


def rebuild(archive, output):
    check(sha(archive) == ARCHIVE_SHA, 'Released archive fingerprint differs')
    signed = read(ROOT/'research/b_local_components_1123/protocol.json')
    with zipfile.ZipFile(archive) as source, source.open(MEMBER) as stream:
        reader = csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig', newline=''))
        rows = [(r['headline_A'], r['headline_B'], int(r['label_pairwise'])) for r in reader]
    task = SimpleNamespace(kind='pair_text', X_train=[r[:2] for r in rows],
                           y_train=np.array([r[2] for r in rows]))
    training, reserve, _ = grouped_partition(task, 41101, 'shared_text')
    training, reserve = sorted(map(int, training)), sorted(map(int, reserve))
    groups = sample_groups(task, 'shared_text')
    check(len(training)==signed['rows'] and len(reserve)==signed['holdout'] and
          set(groups[training]).isdisjoint(groups[reserve]), 'Group split differs')
    output.mkdir(parents=True, exist_ok=False)
    packets(rows, training, True, output/'anonymous.zip')
    packets(rows, reserve, False, output/'features.zip')
    save(output/'scoring_only.json', dict(labels=[rows[i][2] for i in reserve], original_rows=reserve))
    save(output/'groups_only.json', list(groups[reserve]))
    for name,key in (('anonymous.zip','training_sha'),('features.zip','feature_sha'),
                     ('scoring_only.json','locked_scores'),('groups_only.json','locked_groups')):
        check(sha(output/name) == signed[key], 'Rebuilt artifact differs: '+name)
    save(output/'protocol.json', signed)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    rebuild(args.archive,args.output)
