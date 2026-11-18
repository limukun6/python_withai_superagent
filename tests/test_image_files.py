import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
import warnings
from unittest.mock import patch,MagicMock
import zipfile

import numpy as np
from PIL import Image

from superagent.data import load_development_task,load_training_task,load_prediction_features
from superagent.image_files import image_contract,load_image_rows,MAX_SOURCE_PIXELS


def encoded(image,format='PNG',**kwargs):
    stream=io.BytesIO();image.save(stream,format=format,**kwargs);return stream.getvalue()


def fixture(path,modify=None,extra=None):
    manifest={'schema_version':1,'modality':'image','objective':'classification','metric':'accuracy',
        'columns':[{'name':'image','type':'image_path'}],'target':'label',
        'image_preprocessing':{'width':4,'height':3,'mode':'RGB','resize':'stretch_bilinear'},
        'splits':{'train':{'format':'csv','path':'train.csv'},'validation':{'format':'csv','path':'val.csv'},
            'test':{'format':'csv','path':'test.csv'}}}
    contents={'train.csv':b'image,label\nimages/blue.png,blue\nimages/red.jpg,red\n',
        'val.csv':b'image,label\nimages/gray.png,blue\nimages/red.jpg,red\n',
        'test.csv':b'image\nimages/gray.png\nimages/blue.png\n',
        'images/blue.png':encoded(Image.new('RGB',(7,5),(0,0,255))),
        'images/red.jpg':encoded(Image.new('RGB',(11,8),(255,0,0)),format='JPEG'),
        'images/gray.png':encoded(Image.new('L',(2,9),93)),
        'test_labels.json':b'NEVER READ THIS LABEL FILE'}
    if modify:modify(manifest,contents)
    with zipfile.ZipFile(path,'w') as archive:
        archive.writestr('packet/manifest.json',json.dumps(manifest))
        for name,content in contents.items():archive.writestr('packet/'+name,content)
        if extra:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore',UserWarning)
                for name,content in extra:archive.writestr('packet/'+name,content)
    return manifest


class ImageFileTests(unittest.TestCase):
    def test_different_sizes_modes_and_formats_preserve_csv_order_labels_and_uint8_shape(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'images.zip';fixture(path);task=load_development_task(path)
        self.assertEqual(task.kind,'image');self.assertEqual(task.X_train.shape,(2,3,4,3));self.assertEqual(task.X_train.dtype,np.uint8)
        self.assertEqual(task.y_train.tolist(),['blue','red'])
        np.testing.assert_array_equal(task.X_train[0],np.broadcast_to([0,0,255],(3,4,3)))
        self.assertGreater(task.X_train[1,:,:,0].min(),250)
        np.testing.assert_array_equal(task.X_val[0],np.full((3,4,3),93))
        self.assertEqual(task.spec['feature_shape'],[3,4,3]);self.assertIn('image_decoder_version',task.spec)

    def test_grayscale_output_and_transparency_use_fixed_white_background(self):
        def edit(manifest,contents):
            contents['images/blue.png']=encoded(Image.new('RGBA',(5,5),(255,0,0,0)))
            manifest['image_preprocessing']['mode']='L'
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'images.zip';fixture(path,edit);task=load_training_task(path)
        self.assertEqual(task.X_train.shape,(2,3,4));self.assertTrue(np.all(task.X_train[0]==255))
        def semi(manifest,contents):contents['images/blue.png']=encoded(Image.new('RGBA',(4,3),(0,0,255,128)))
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'images.zip';fixture(path,semi);task=load_training_task(path)
        np.testing.assert_array_equal(task.X_train[0,0,0],[127,127,255])

    def test_exif_orientation_is_applied_before_declared_resize(self):
        values=np.arange(4*6*3,dtype=np.uint8).reshape(4,6,3)
        image=Image.fromarray(values);exif=image.getexif();exif[274]=6
        def edit(manifest,contents):
            manifest['image_preprocessing'].update(width=4,height=6)
            contents['images/blue.png']=encoded(image,exif=exif)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'images.zip';fixture(path,edit);task=load_training_task(path)
        np.testing.assert_array_equal(task.X_train[0],np.rot90(values,k=-1))

    def test_train_only_and_development_do_not_open_prediction_or_decoy_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'images.zip';fixture(path);original=zipfile.ZipFile.read;opened=[]
            def traced(archive,member,*args,**kwargs):
                name=member.filename if isinstance(member,zipfile.ZipInfo) else str(member);opened.append(name)
                if 'test' in name:raise AssertionError('Development cannot read prediction members')
                return original(archive,member,*args,**kwargs)
            with patch.object(zipfile.ZipFile,'read',new=traced):
                load_training_task(path);self.assertNotIn('packet/val.csv',opened)
                opened.clear();load_development_task(path)
            self.assertIn('packet/val.csv',opened);self.assertNotIn('packet/test.csv',opened)

    def test_prediction_uses_only_requested_images_and_frozen_contract_without_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'images.zip';fixture(path);task=load_development_task(path);original=zipfile.ZipFile.read;opened=[]
            def traced(archive,member,*args,**kwargs):
                name=member.filename if isinstance(member,zipfile.ZipInfo) else str(member);opened.append(name)
                if name.endswith(('train.csv','val.csv','test_labels.json')):raise AssertionError('Prediction cannot read training or label members')
                return original(archive,member,*args,**kwargs)
            with patch.object(zipfile.ZipFile,'read',new=traced):values=load_prediction_features(path,'image',task.spec)
        self.assertEqual(values.shape,(2,3,4,3));self.assertTrue(np.all(values[0]==93))
        self.assertEqual(set(opened),{'packet/manifest.json','packet/test.csv','packet/images/gray.png','packet/images/blue.png'})

    def test_prediction_contract_changes_fail_before_any_prediction_images_are_read(self):
        with tempfile.TemporaryDirectory() as temp:
            base=Path(temp)/'base.zip';fixture(base);task=load_development_task(base)
            for mode in ('resize','mode','width','decoder','version','format'):
                path=Path(temp)/(mode+'.zip');spec=copy.deepcopy(task.spec)
                def edit(manifest,contents):
                    if mode=='format':manifest['splits']['test']['format']='npz'
                    elif mode=='width':manifest['image_preprocessing']['width']=5
                    elif mode=='mode':manifest['image_preprocessing']['mode']='L'
                    elif mode=='resize':manifest['image_preprocessing']['resize']='nearest'
                fixture(path,edit)
                if mode=='decoder':spec['image_decoder_policy']='different-policy'
                if mode=='version':spec['image_decoder_version']='different-version'
                original=zipfile.ZipFile.read;opened=[]
                def traced(archive,member,*args,**kwargs):opened.append(str(member));return original(archive,member,*args,**kwargs)
                with self.subTest(mode=mode),patch.object(zipfile.ZipFile,'read',new=traced),self.assertRaises(ValueError):
                    load_prediction_features(path,'image',spec)
                self.assertEqual(opened,['packet/manifest.json'])

    def test_paths_cannot_escape_point_to_test_members_or_resolve_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            for index,raw in enumerate(('/outside.png','../outside.png','C:\\outside.png','images/test_case.png','missing.png')):
                path=Path(temp)/f'bad_{index}.zip'
                def edit(manifest,contents):contents['train.csv']=f'image,label\n{raw},blue\nimages/red.jpg,red\n'.encode()
                fixture(path,edit)
                with self.subTest(path=raw),self.assertRaises(ValueError):load_training_task(path)
            path=Path(temp)/'duplicate.zip';fixture(path,extra=[('images/blue.png',encoded(Image.new('RGB',(2,2))))])
            with self.assertRaisesRegex(ValueError,'ambiguous'):load_training_task(path)

    def test_missing_targets_bad_headers_unsupported_depth_and_corrupt_or_mislabeled_images_fail(self):
        edits={
            'missing_target':lambda m,c:c.update({'train.csv':b'image\nimages/blue.png\n'}),
            'wrong_type':lambda m,c:m['columns'][0].update(type='text'),
            'corrupt':lambda m,c:c.update({'images/blue.png':b'not an image'}),
            'mislabeled':lambda m,c:c.update({'images/blue.png':encoded(Image.new('RGB',(2,2)),format='JPEG')}),
            'sixteen_bit':lambda m,c:c.update({'images/blue.png':encoded(Image.fromarray(np.ones((3,4),dtype=np.uint16)*3000))}),
        }
        with tempfile.TemporaryDirectory() as temp:
            for name,edit in edits.items():
                path=Path(temp)/(name+'.zip');fixture(path,edit)
                with self.subTest(case=name),self.assertRaises((ValueError,OSError)):load_training_task(path)

    def test_invalid_preprocessing_and_size_limits_fail_without_large_allocations(self):
        base={'width':4,'height':3,'mode':'RGB','resize':'stretch_bilinear'}
        for changes in ({'width':True},{'height':0},{'height':513},{'mode':'RGBA'},{'resize':'nearest'},{'extra':'ignored'}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):image_contract({**base,**changes})
        manifest={'image_preprocessing':{**base,'width':512,'height':512},'_manifest_parent':'.'}
        with self.assertRaisesRegex(ValueError,'256 MiB'):load_image_rows(None,manifest,'train',['x.png']*342)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'images.zip';fixture(path)
            source=MagicMock();source.format='PNG';source.width=MAX_SOURCE_PIXELS+1;source.height=1
            context=MagicMock();context.__enter__.return_value=source
            with patch('PIL.Image.open',return_value=context),self.assertRaisesRegex(ValueError,'16-megapixel'):
                load_training_task(path)
            source.load.assert_not_called()


if __name__=='__main__':unittest.main()
