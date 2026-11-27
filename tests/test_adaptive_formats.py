import gzip
from contextlib import closing
import io
import json
from pathlib import Path
import sqlite3
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import pyarrow as pa

from superagent.data import load_prediction_features, load_training_task
from superagent.format_intake import adapt_training, predict_records
from superagent.format_sources import Package

TRAIN = [(i, 'low' if i < 24 else 'high') for i in range(48)]
QUERY = [47, 2, 36, 17]
FIXED_DOC = ('Records are headerless and fixed-width.\n'
             'Positions 1 through 3 hold the field signal.\nPositions 4 through 6 hold the field offset.\n'
             'Positions 7 through 8 hold the field verdict.\nFor scoring, predict verdict.')
FIXED_PLAN = dict(fields=[dict(name=n,start=a,end=b,layout_evidence=i) for i,(n,a,b) in
                 enumerate([('signal',1,3),('offset',4,6),('verdict',7,8)])], target='verdict', goal_evidence=0)


def tar_package(path, files):
    with tarfile.open(path, 'w:gz') as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return path


def nested_package(root):
    train = '\n'.join(json.dumps(dict(sensor=dict(reading=i), label=y)) for i,y in TRAIN).encode()
    query = '\n'.join(json.dumps(dict(sensor=dict(reading=i))) for i in QUERY).encode()
    training = tar_package(root/'p.bundle', {'records/train.payload.gz':gzip.compress(train),
                           'validation/answers.never':b'inaccessible'})
    prediction = tar_package(root/'q.bundle', {'records/test.payload.gz':gzip.compress(query)})
    return training, prediction


def workbook(path):
    from xml.etree.ElementTree import Element, SubElement, tostring
    ns = 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'
    rel = 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
    pkg = 'http://schemas.openxmlformats.org/package/2006/relationships'
    book = Element('workbook', xmlns=ns, attrib={'xmlns:r':rel})
    sheets = SubElement(book,'sheets')
    links = Element('Relationships', xmlns=pkg)
    with zipfile.ZipFile(path, 'w') as z:
        for index, (name, rows) in enumerate([('train',[('signal','label'),*TRAIN]),
                ('test',[('signal',),*((x,) for x in QUERY)]), ('validation',[])], 1):
            SubElement(sheets, 'sheet', name=name, sheetId=str(index), attrib={'r:id':f'r{index}'})
            SubElement(links,'Relationship',Id=f'r{index}',Type=rel+'/worksheet',Target=f'worksheets/sheet{index}.xml')
            sheet = Element('worksheet', xmlns=ns)
            data = SubElement(sheet,'sheetData')
            for n, row in enumerate(rows, 1):
                node = SubElement(data,'row',r=str(n))
                for j,value in enumerate(row):
                    cell = SubElement(node,'c',r=chr(65+j)+str(n),t='inlineStr')
                    SubElement(SubElement(cell,'is'),'t').text = str(value)
            z.writestr(f'xl/worksheets/sheet{index}.xml', tostring(sheet) if rows else b'not valid XML')
        z.writestr('xl/workbook.xml',tostring(book))
        z.writestr('xl/_rels/workbook.xml.rels',tostring(links))
    return path


class AssistantFixture:
    def __init__(self, reply):
        self.reply, self.request_count, self.max_requests, self.spent, self.blocked = reply, 0, 1, 0.0, False
        self.payloads = []
    def set_deadline(self, deadline):
        self.deadline = deadline
    def ask(self, payload, logger, role):
        self.request_count += 1
        self.payloads.append(payload)
        return self.reply


class AdaptiveFormats(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def check(self, training, query, codec):
        task = adapt_training(training)
        self.assertEqual(task.spec['input_recipe']['codec'],codec)
        self.assertEqual(task.y_train.tolist(), [y for _,y in TRAIN])
        np.testing.assert_equal(task.X_train[:,0], [i for i,_ in TRAIN])
        x = load_prediction_features(query, task.kind, json.loads(json.dumps(task.spec)))
        np.testing.assert_equal(x[:,0], QUERY)
        return task

    def test_nested_gzip_tar_and_member_isolation(self):
        training, query = nested_package(self.root)
        opened = []
        original = Package.read
        def audited(archive,name,*args):
            opened.append(name)
            self.assertNotIn('validation',name)
            return original(archive,name,*args)
        with patch.object(Package, 'read', audited):
            self.check(training, query, 'json-lines')
        self.assertIn('records/train.payload.gz',opened)

    def test_workbook_never_opens_other_sheets_during_training(self):
        path = workbook(self.root/'anonymous.bin')
        opened = []
        original = zipfile.ZipFile.open
        def audit(archive, name, *args, **kw):
            opened.append(name.filename if isinstance(name,zipfile.ZipInfo) else name)
            return original(archive,name,*args,**kw)
        with patch.object(zipfile.ZipFile,'open',audit):
            task = adapt_training(path)
        self.assertIn('xl/worksheets/sheet1.xml',opened)
        self.assertNotIn('xl/worksheets/sheet2.xml',opened)
        self.assertNotIn('xl/worksheets/sheet3.xml',opened)
        np.testing.assert_equal(predict_records(path,task.kind,task.spec)[:,0],QUERY)

    def test_sqlite(self):
        path = self.root/'anonymous.bin'
        with closing(sqlite3.connect(path)) as db, db:
            db.execute('CREATE TABLE train(signal INTEGER,label TEXT)')
            db.execute('CREATE TABLE test(signal INTEGER)')
            db.executemany('INSERT INTO train VALUES(?,?)',TRAIN)
            db.executemany('INSERT INTO test VALUES(?)',[(i,) for i in QUERY])
            db.execute('CREATE VIEW validation AS SELECT nonsense FROM absent')
        self.check(path,path,'sqlite')

    def test_utf16_to_json(self):
        training, query = self.root/'train.dat',self.root/'test.json'
        training.write_bytes(('signal|label\n'+''.join(f'{i}|{y}\n' for i,y in TRAIN)).encode('utf-16'))
        query.write_text(json.dumps([dict(signal=x) for x in QUERY]))
        self.check(training,query,'delimited')

    def test_arrow_xml_and_arff(self):
        for fmt in ('arrow-ipc','xml-records','arff'):
            with self.subTest(fmt=fmt):
                paths = []
                for role,rows in [('train',TRAIN),('test',[(i,None) for i in QUERY])]:
                    path = self.root/(role+'.blob')
                    if fmt == 'arrow-ipc':
                        columns = dict(signal=[i for i,_ in rows])
                        if role == 'train':
                            columns['label'] = [y for _,y in rows]
                        arr = pa.table(columns)
                        with pa.OSFile(str(path),'wb') as stream, pa.ipc.new_file(stream,arr.schema) as handle:
                            handle.write_table(arr)
                    elif fmt == 'xml-records':
                        rows_xml = [f'<row><signal>{i}</signal>'+(f'<label>{y}</label>' if y else '')+'</row>' for i,y in rows]
                        path.write_text('<dataset>'+''.join(rows_xml)+'</dataset>')
                    else:
                        path.write_text('@relation anonymous\n@attribute signal numeric\n'+
                            ('@attribute label {low,high}\n' if role=='train' else '')+'@data\n'+
                            ''.join(str(i)+(f',{y}' if y else '')+'\n' for i,y in rows))
                    paths.append(path)
                self.check(*paths,fmt)

    def test_mixed_rows_filtered_before_label_inspection(self):
        path = self.root/'records.json'
        path.write_text(json.dumps([dict(split='test',signal=x,label=['withheld']) for x in QUERY]+
                                  [dict(split='train',signal=i,label=y) for i,y in TRAIN]))
        self.check(path,path,'json-records')

    def test_manifest_custom_split_markers(self):
        package = self.root/'bound.zip'
        manifest = dict(schema_version=1, modality='tabular', target='outcome', metric='accuracy',
            columns=[dict(name='signal',type='numeric')], split_column='bucket', split_values={'train':'L','test':'E'},
            splits={s:dict(format='json',path='all.dat') for s in ('train','test')})
        records = [dict(bucket='L',signal=i,outcome=y) for i,y in TRAIN]+[dict(bucket='E',signal=i) for i in QUERY]
        with zipfile.ZipFile(package,'w') as z:
            z.writestr('manifest.json',json.dumps(manifest))
            z.writestr('all.dat',json.dumps(records))
        self.check(package,package,'json-records')

    def test_fixed_width_checked_plan_replays_without_a_model(self):
        (self.root/'README.md').write_text(FIXED_DOC)
        training, query = self.root/'train.dat',self.root/'test.dat'
        training.write_text('002004 N\n007014 Y\n009018 Y\n')
        query.write_text('007014\n002004\n')
        client = AssistantFixture(FIXED_PLAN)
        task = adapt_training(training,client)
        self.assertEqual(client.request_count,1)
        self.assertNotIn('007014',json.dumps(client.payloads))
        np.testing.assert_equal(predict_records(query,task.kind,task.spec),[[7,14],[2,4]])
        wrong = json.loads(json.dumps(FIXED_PLAN))
        wrong['fields'][0]['end'] = 8
        with self.assertRaises(ValueError):
            adapt_training(training,AssistantFixture(wrong))

    def test_ai_links_goal_to_field_dictionary(self):
        (self.root/'README.md').write_text('Predict the final referee ruling.\nThe verdict field stores the final referee ruling.\nThe signal field records a sensor reading.')
        p = self.root/'train.data'
        p.write_text(json.dumps([dict(signal=i,verdict=y) for i,y in TRAIN]))
        proposal = dict(target='verdict',goal_evidence=0,field_evidence=0,link='referee ruling')
        client = AssistantFixture(proposal)
        self.assertEqual(adapt_training(p,client).spec['input_recipe']['target'],'verdict')
        self.assertEqual(client.request_count,1)
        with self.assertRaises(ValueError):
            adapt_training(p,AssistantFixture(dict(proposal,target='signal')))

    def test_opaque_ambiguous_and_heldout(self):
        p = self.root/'train.blob'
        for raw in (bytes([0,1,254,255]),b'{"a":2,"b":3}\n{"a":3,"b":4}',
                    b'[{"split":"Q","signal":1,"label":"a"}]'):
            p.write_bytes(raw)
            with self.assertRaises(ValueError):
                adapt_training(p)
        other = self.root/'test.csv'
        other.write_text('signal,label\n1,a\n2,b')
        with self.assertRaises(ValueError):
            load_training_task(other,read_declarations=True)
        with self.assertRaises(ValueError):
            adapt_training(other)

    def test_frozen_feature_drift_and_training_reuse(self):
        p = self.root/'batch.dat'
        p.write_text(json.dumps([dict(signal=i,label=y) for i,y in TRAIN]))
        task = adapt_training(p)
        with self.assertRaises(ValueError):
            predict_records(p,task.kind,task.spec)
        q = self.root/'test.dat'
        q.write_text('[{"other":1}]')
        with self.assertRaises(ValueError):
            predict_records(q,task.kind,task.spec)

    def test_new_static_bitmap_codecs_in_tar_match_pixels(self):
        from PIL import Image
        colors = [(20,30,40),(75,100,125)]
        for fmt,ext in [('PNG','png'),('BMP','bmp'),('TIFF','tiff'),('WEBP','webp'),('GIF','gif')]:
            with self.subTest(fmt=fmt):
                files = {}
                for i,color in enumerate(colors):
                    buf = io.BytesIO()
                    Image.new('RGB',(7,8),color).save(buf,format=fmt,**({'lossless':True} if fmt=='WEBP' else {}))
                    files[f'train/c{i}/sample.{ext}'] = buf.getvalue()
                p = tar_package(self.root/(ext+'.tar.gz'),files)
                task = load_training_task(p,read_declarations=True)
                self.assertEqual(task.kind,'image')
                self.assertEqual(task.y_train.tolist(),['c0','c1'])
                np.testing.assert_equal(task.X_train[:,0,0,:],colors)

    def test_animated_bitmap_and_legacy_object_cells(self):
        from datetime import date
        from PIL import Image
        from superagent.legacy_array import load_array
        buf = io.BytesIO()
        seq = [Image.new('RGB',(4,4),c) for c in ('red','blue')]
        seq[0].save(buf,format='GIF',save_all=True,append_images=seq[1:])
        path = tar_package(self.root/'image.tar.gz',{'train/a/one.gif':buf.getvalue(),'train/b/two.gif':buf.getvalue()})
        with self.assertRaisesRegex(ValueError,'multiframe'):
            load_training_task(path,read_declarations=True)
        for array in [np.array([['a',None],['b',3]],dtype=object),
                      np.asfortranarray(np.array([['a','b'],['c','d']],dtype=object))]:
            buf = io.BytesIO()
            np.save(buf,array)
            np.testing.assert_equal(load_array(buf.getvalue()),array)
        buf = io.BytesIO()
        np.save(buf,np.array([date(2026,1,1)],dtype=object))
        with self.assertRaises(ValueError):
            load_array(buf.getvalue())

    def test_partial_with_known_labels_and_unusable_structured_features(self):
        from superagent.autonomous_run import train
        from superagent.predict import create_predictions
        p, q = self.root/'train.dat', self.root/'test.dat'
        p.write_text(json.dumps([dict(signal=[1,2],label=y) for y in ('yes','yes','no')]))
        q.write_text('[{"signal":[4,5]},{"signal":[8,9]}]')
        run = self.root/'run'
        self.assertEqual(train(p,run,fit_limit=2),0)
        out = json.loads((run/'run_outcome.json').read_text())
        self.assertEqual((out['status'],out['attempted_classifier_fits']),('partial',0))
        dest = create_predictions(q,run,self.root/'out'/'predictions.csv')
        import csv
        with dest.open() as stream:
            self.assertEqual([r['prediction'] for r in csv.DictReader(stream)],['yes','yes'])

    def test_unresolved_assistance_is_counted_on_input_failure(self):
        from superagent.autonomous_run import train
        class Unresolved(AssistantFixture):
            def ask(self,payload,logger,role):
                self.request_count += 1
                self.blocked = True
                logger.usage(event='unresolved_call',role=role,usage_known=False,course_cost_rmb=None)
                raise RuntimeError('synthetic accounting interruption')
        (self.root/'README.md').write_text(FIXED_DOC)
        p=self.root/'train.dat'
        p.write_text('001002 N\n007014 Y\n')
        run=self.root/'run'
        self.assertEqual(train(p,run,fit_limit=2,flash=Unresolved(None)),2)
        out=json.loads((run/'run_outcome.json').read_text())
        self.assertEqual((out['new_api_requests'],out['unresolved_api_requests'],out['attempted_classifier_fits']),(1,1,0))
        self.assertIsNone(out['complete_run_course_cost_rmb'])

    def test_nested_split_and_contradictory_manifest(self):
        p=self.root/'rows.dat'
        p.write_text(json.dumps([dict(meta=dict(split='test'),signal=x,label=['private']) for x in QUERY]+
                                [dict(meta=dict(split='train'),signal=i,label=y) for i,y in TRAIN]))
        self.check(p,p,'json-records')
        p=self.root/'p.zip'
        with zipfile.ZipFile(p,'w') as z:
            z.writestr('manifest.json',json.dumps(dict(schema_version=1,modality='tabular',target='label',
                columns=[dict(name='signal',type='numeric')],splits={'train':dict(format='xlsx',path='train.dat')})))
            z.writestr('train.dat','[{"signal":1,"label":"a"},{"signal":2,"label":"b"}]')
        with self.assertRaisesRegex(ValueError,'format conflicts'):
            adapt_training(p)

    def test_standard_fixed_width_and_unusual_documented_separator_are_local(self):
        docs, p, q = self.root/'README.md',self.root/'train.dat',self.root/'test.dat'
        docs.write_text('Headerless fixed-width.\nsignal: 1-3\noffset: 4-6\nverdict: 7-8\nPredict verdict.')
        p.write_text('002004 N\n007014 Y\n009018 Y\n')
        q.write_text('007014\n002004\n')
        unused = AssistantFixture(None)
        task = adapt_training(p,unused)
        np.testing.assert_equal(predict_records(q,task.kind,task.spec),[[7,14],[2,4]])
        self.assertEqual(unused.request_count,0)
        docs.write_text('The separator is "§".', encoding='utf-8')
        p.write_text('signal§label\n'+''.join(f'{i}§{y}\n' for i,y in TRAIN), encoding='utf-8')
        q.write_text('signal\n'+'\n'.join(map(str,QUERY))+'\n', encoding='utf-8')
        self.check(p,q,'delimited')

    def test_missing_or_contradictory_explicit_manifest_stays_unavailable(self):
        from superagent.autonomous_run import train
        for i, value in enumerate([[], {'schema_version':1,'modality':'tabular','splits':[]},
                {'schema_version':99,'modality':'tabular','splits':{}}]):
            p = self.root/(str(i)+'.zip')
            with zipfile.ZipFile(p,'w') as archive:
                archive.writestr('manifest.json',json.dumps(value))
                archive.writestr('train.dat','[{"signal":1,"label":"a"},{"signal":2,"label":"b"}]')
            run = self.root/('run'+str(i))
            self.assertEqual(train(p,run,fit_limit=2),2)
            self.assertFalse((run/'selected_model.joblib').exists())
            self.assertEqual(json.loads((run/'run_outcome.json').read_text())['delivery_status'],'unavailable')
