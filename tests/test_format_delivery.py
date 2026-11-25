import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from superagent.data import load_training_task, load_prediction_features
from superagent.predict import create_predictions
from unified_agent import train


def zipped(path, entries):
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name,data in entries.items():
            z.writestr(name,data)
    return path


def as_npy(values):
    out=io.BytesIO()
    np.save(out,np.asarray(values),allow_pickle=False)
    return out.getvalue()


def color_bytes(i):
    from PIL import Image
    out=io.BytesIO()
    Image.new('RGB',(6,6),(i,0,0)).save(out,format='PNG')
    return out.getvalue()


def fixtures():
    import pyarrow as pa
    import pyarrow.parquet as pq
    features=[float(i%7) for i in range(90)]
    labels=['red','green','blue']*30
    def parquet(table):
        out=io.BytesIO()
        pq.write_table(pa.table(table),out)
        return out.getvalue()
    readme='The label column is verdict. The primary metric is macro F1.'
    table={'README.md':readme,'train.csv':'amount,verdict\n'+''.join(f'{i},{labels[i]}\n' for i in range(90))}
    split={'test.csv':'amount\n4\n3\n9\n'}
    columns=[dict(name='amount',type='numeric')]
    manifest=dict(schema_version=1,objective='classification',modality='tabular',metric='macro_f1',columns=columns,target='verdict')
    standard={**manifest,'splits':{'train':{'format':'csv','path':'learn.csv'}}}
    held={**manifest,'splits':{'test':{'format':'csv','path':'features.csv'}}}
    image_train={'training/'+label+'/'+str(i)+'.png':color_bytes(i*2) for i,label in enumerate(labels)}
    image_held={'test/images/'+str(i)+'.png':color_bytes(i*3) for i in range(3)}
    class_first={label+'/training/'+str(i)+'.png':color_bytes(i*2) for i,label in enumerate(labels)}
    arrays=io.BytesIO()
    np.savez(arrays,X=np.arange(180).reshape(90,2),y=np.arange(90)%3)
    queries=io.BytesIO()
    np.savez(queries,X=np.arange(6).reshape(3,2))
    pairs=[(f'Here is the first item number {i}',f'An alternate item with number {i}') for i in range(80)]
    pair_text='first_text,second_text,choice\n'+''.join(f'{l},{r},{i%2}\n' for i,(l,r) in enumerate(pairs))
    return {
        'plain_csv':(table,split,'numeric_tabular','macro_f1'),
        'manifest_csv':({'manifest.json':json.dumps(standard),'learn.csv':table['train.csv']},
                        {'manifest.json':json.dumps(held),'features.csv':split['test.csv']},'numeric_tabular','macro_f1'),
        'tsv':({'README.md':readme,'train.tsv':table['train.csv'].replace(',','\t')},
               {'test.tsv':split['test.csv']},'numeric_tabular','macro_f1'),
        'parquet':({'README.md':readme,'train.parquet':parquet(dict(amount=features,verdict=labels))},
                   {'test.parquet':parquet(dict(amount=[4.,3.,9.]))},'numeric_tabular','macro_f1'),
        'folder':(image_train,image_held,'image','accuracy'),
        'class_first_folder':(class_first,{'bird/testing/'+str(i)+'.png':color_bytes(i*4) for i in range(3)},
                              'image','accuracy'),
        'declared_paths':({'README.md':readme+' Training file is learn.csv. Prediction file is query.csv.',
                           'contest/learn.csv':table['train.csv']},{'query.csv':split['test.csv']},'numeric_tabular','macro_f1'),
        'array':({'train.npz':arrays.getvalue()},{'test.npz':queries.getvalue()},'numeric_tabular','accuracy'),
        'paired':({'train.csv':pair_text,'README.md':'Target column is choice. Label 1 means first wins and 0 means second wins.'},
                  {'test.csv':'first_text,second_text\n'+''.join(f'{l},{r}\n' for l,r in pairs[:3])},'pair_text','accuracy'),
    }


class FormatDeliveryTests(unittest.TestCase):
    def test_discovery_no_task_id_and_frozen_prediction(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name,(training,features,kind,metric) in fixtures().items():
                with self.subTest(name=name):
                    package=zipped(root/(name+'.zip'),{**training,'validation/dont-read':b'x','test/labels.secret':b'x'})
                    guard=zipfile.ZipFile.read
                    def read(z,member,*args,**kwargs):
                        if str(member) not in training:
                            raise AssertionError('Opened nontraining member: '+str(member))
                        return guard(z,member,*args,**kwargs)
                    with patch.object(zipfile.ZipFile,'read',read):
                        task=load_training_task(package,read_declarations=True)
                    self.assertEqual((task.kind,task.spec['metric']),(kind,metric))
                    self.assertEqual(len(np.unique(task.y_train)),2 if kind=='pair_text' else 3)
                    test=zipped(root/(name+'-blind.zip'),features)
                    self.assertEqual(len(load_prediction_features(test,kind,task.spec)),3)

    def test_duplicate_targets_are_not_resolved_from_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            package=zipped(Path(tmp)/'raw.zip',{'README.md':'Target column is foo. Target column is bar.',
                         'train.csv':'foo,bar\ncat,f\ndog,g\n'})
            with self.assertRaises(ValueError):
                load_training_task(package,read_declarations=True)

    def test_standalone_and_directory_preserve_explicit_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'README.md').write_text('Training file is measurements.tsv. Prediction file is future.tsv. 标签列为 verdict。主指标为 macro F1。')
            (root/'measurements.tsv').write_text('amount\tverdict\n1\tb\n2\ta\n3\tb\n')
            for source in (root,root/'measurements.tsv'):
                task=load_training_task(source,read_declarations=True)
                self.assertEqual(task.spec['metric'],'macro_f1')
                (root/'future.tsv').write_text('amount\n8\n9\n')
                self.assertEqual(len(load_prediction_features(root/'future.tsv',task.kind,task.spec)),2)

    def test_partial_baseline_needs_real_label_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            pkg=zipped(root/'raw.zip',{'README.md':'Target column is verdict.',
                 'train.csv':'broken,verdict\n,yes\n,yes\n,no\n','test.csv':b'FORBIDDEN DURING TRAINING'})
            run=root/'run'
            self.assertEqual(train(pkg,run),0)
            outcome=json.loads((run/'agent/run_outcome.json').read_text())
            self.assertTrue(outcome['contract_unresolved'])
            self.assertEqual((outcome['delivery_status'],outcome['attempted_classifier_fits']),('partial',0))
            test=zipped(root/'blind.zip',{'test.csv':'broken\na\nb\nc\n'})
            file=create_predictions(test,run/'agent',root/'pred/predictions.csv')
            with file.open() as f:
                self.assertEqual([r['prediction'] for r in csv.DictReader(f)],['yes']*3)
            self.assertEqual(json.loads((file.parent/'prediction_manifest.json').read_text())['delivery_status'],'partial')

    def test_independent_training_labels_survive_missing_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manifest=dict(schema_version=1,modality='tabular',metric='accuracy',
                          splits={'train':dict(format='npy',path='missing.npy',target_path='train_labels.npy'),
                                  'test':dict(format='npy',path='query.npy')})
            pkg=zipped(root/'raw.zip',{'manifest.json':json.dumps(manifest),'train_labels.npy':as_npy([3,2,3,3])})
            run=root/'run'
            self.assertEqual(train(pkg,run),0)
            out=create_predictions(zipped(root/'blind.zip',{'manifest.json':json.dumps(manifest),'query.npy':as_npy([[1],[0]])}),
                                   run/'agent',root/'pred/predictions.csv')
            with out.open() as f:
                self.assertEqual([r['prediction'] for r in csv.DictReader(f)],['3','3'])

    def test_unknown_domain_preserves_failure_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            pkg=zipped(root/'raw.zip',{'train.csv':'a,b\nfoo,red\nbar,green\n','test.csv':b'FORBIDDEN'})
            self.assertNotEqual(train(pkg,root/'run'),0)
            outcome=json.loads((root/'run/agent/run_outcome.json').read_text())
            self.assertTrue(outcome['contract_unresolved'])
            self.assertEqual(outcome['delivery_status'],'unavailable')
            self.assertFalse((root/'run/agent/final_selection.json').exists())
