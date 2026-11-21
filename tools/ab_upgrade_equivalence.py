"""Migrate a legacy B symmetric CSV packet using an explicit mapping for both source labels."""
import argparse
import csv
import io
import json
from pathlib import Path
import zipfile


def upgrade(source, target, negative, positive):
    if not negative or not positive or negative == positive:
        raise ValueError('Two distinct source labels are required')
    with zipfile.ZipFile(source) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):
            raise ValueError('Duplicate packet member')
        manifest = json.loads(archive.read('manifest.json'))
        if (manifest.get('modality')!='pair_text' or manifest.get('pair_semantics')!='symmetric' or
                manifest.get('pair_label_one') is not None or len(manifest['splits'])!=1):
            raise ValueError('Expected a legacy symmetric, single-split B packet')
        split, spec = next(iter(manifest['splits'].items()))
        if spec['format']!='csv' or split not in ('train','test') or spec['path']=='manifest.json':
            raise ValueError('Expected a training CSV or an unlabelled feature CSV')
        data = archive.read(spec['path'])
        reader = csv.DictReader(io.StringIO(data.decode('utf-8-sig'),newline=''))
        features = [c['name'] for c in manifest['columns']]
        label = manifest['target']
        expected = features + ([label] if split=='train' else [])
        if reader.fieldnames!=expected:
            raise ValueError('The source columns differ from the registered role')
        mapping = {negative:0,positive:1}
        if 'positive_label' in manifest:
            if str(manifest['positive_label'])!=positive:
                raise ValueError('The supplied positive label contradicts the manifest')
            manifest['positive_label']=1
        if split=='train':
            converted = io.StringIO(newline='')
            writer = csv.DictWriter(converted,fieldnames=expected,lineterminator='\n')
            writer.writeheader()
            for row in reader:
                if row[label] not in mapping:
                    raise ValueError('Unmapped source class in legacy training packet')
                row[label] = mapping[row[label]]
                writer.writerow(row)
            data = converted.getvalue().encode()
        manifest['pair_label_one'] = 'equivalent'
        updated = {'manifest.json':json.dumps(manifest,sort_keys=True).encode(),spec['path']:data}
        with zipfile.ZipFile(target,'x') as migrated:
            for member in archive.infolist():
                migrated.writestr(member,updated[member.filename] if member.filename in updated else archive.read(member))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--negative',required=True)
    p.add_argument('--positive',required=True)
    a=p.parse_args()
    upgrade(a.source,a.output,a.negative,a.positive)
