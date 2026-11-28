import csv
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from tools.evaluate_frozen_released import feature_packet, prediction_rows


class OfficialInputTests(unittest.TestCase):
    def test_projection_preserves_row_order_and_never_passes_the_label_to_the_text_predictor(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            features=[('A, with comma','second'),('A\ncontinued','第三')]
            for index,labels in enumerate([['0','1'],['1','0']]):
                source=root/f'source{index}.zip'
                text=io.StringIO(newline='')
                writer=csv.writer(text)
                writer.writerow(['headline_A','headline_B','label_pairwise'])
                writer.writerows([*pair,y] for pair,y in zip(features,labels))
                with zipfile.ZipFile(source,'w') as z:
                    z.writestr('released/test.csv',text.getvalue().encode())
                self.assertEqual(feature_packet(source,'titles',root/f'packet{index}.zip'),2)
            self.assertEqual((root/'packet0.zip').read_bytes(),(root/'packet1.zip').read_bytes())
            with zipfile.ZipFile(root/'packet0.zip') as z:
                exported=list(csv.reader(io.StringIO(z.read('test.csv').decode(),newline='')))
            self.assertEqual(exported,[['headline_A','headline_B'],*[list(pair) for pair in features]])

    def test_table_features_are_copied_verbatim_and_partial_prediction_files_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            numeric=io.BytesIO()
            np.save(numeric,np.arange(6).reshape(3,2))
            with zipfile.ZipFile(root/'source.zip','w') as z:
                z.writestr('data/C_test.npy',numeric.getvalue())
                z.writestr('data/y_test.npy',b'not to be opened during feature projection')
            self.assertEqual(feature_packet(root/'source.zip','table',root/'packet.zip'),3)
            with zipfile.ZipFile(root/'packet.zip') as z:
                self.assertEqual(z.namelist(),['C_test.npy'])
                self.assertEqual(z.read('C_test.npy'),numeric.getvalue())
            pred=root/'predictions.csv'
            pred.write_text('row_id,prediction\n0,1\n2,0\n',encoding='utf-8')
            with self.assertRaises(ValueError):
                prediction_rows(pred,3)


class PairedOfficialScoreTests(unittest.TestCase):
    def test_all_four_seals_precede_truth_and_scores_count_gains_as_well_as_losses(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        from scripts.unified_released_study import read
        from tools.evaluate_frozen_released import ARMS, CASES, digest, labels_from_archive, predict, score

        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            packages=root/'synthetic'
            packages.mkdir()
            inputs=io.BytesIO()
            labels=io.BytesIO()
            np.save(inputs,np.arange(6).reshape(3,2))
            np.save(labels,[0,1,1])
            with zipfile.ZipFile(packages/'Released_Task1-example.zip','w') as z:
                z.writestr('data/C_test.npy',inputs.getvalue())
                z.writestr('data/y_test.npy',labels.getvalue())
            with zipfile.ZipFile(packages/'Released_Task2-example.zip','w') as z:
                z.writestr('data/test.csv','headline_A,headline_B,label_pairwise\none,two,0\nl,r,1\nthird,fourth,0\n')
            refs={}
            bundles=[]
            for arm,values in zip(ARMS,([0,1,0],[1,1,1])):
                bundle=root/(arm+'.zip')
                bundles.append(bundle)
                lines='row_id,prediction\n'+''.join(f'{i},{v}\n' for i,v in enumerate(values))
                with zipfile.ZipFile(bundle,'w') as z:
                    for name,_ in CASES:
                        z.writestr(name+'/solution/bundle_manifest.json',json.dumps(dict(model_sha256=name+arm)))
                        z.writestr(name+'/predict.py','import pathlib,sys\npathlib.Path(sys.argv[-1]).write_text('+repr(lines)+')\n')
                refs[arm]=dict(archive_sha256=digest(bundle),verified={n:dict(selected_model=n+arm) for n,_ in CASES})
            args=SimpleNamespace(bundle=bundles[0],augmented_bundle=bundles[1],out=root/'output',
                                 packages=packages,frozen_commit='a'*40)
            release=dict(solver='test',primary_arm=ARMS[0],arms=refs)
            scorer='tools.evaluate_frozen_released.labels_from_archive'
            with patch(scorer,side_effect=AssertionError('Opened targets during prediction')):
                with self.assertRaises(FileNotFoundError):
                    score(args,release)
                predict(args,release)
                predict(args,release)
                path=args.out/'predictions_sealed.json'
                original=path.read_bytes()
                partial=read(path)
                del partial['cases']['Task2'][ARMS[1]]
                path.write_text(json.dumps(partial))
                with self.assertRaises(ValueError):
                    score(args,release)
                path.write_bytes(original)
            with patch(scorer,wraps=labels_from_archive) as reads:
                score(args,release)
                score(args,release)
                self.assertEqual(reads.call_count,2)
            result=read(args.out/'official_scores.json')['results']
            self.assertEqual(result['Task1']['paired']['corrected_by_adding_validation'],1)
            self.assertEqual(result['Task1']['paired']['regressed_after_adding_validation'],1)
            self.assertEqual(result['Task2']['paired']['regressed_after_adding_validation'],2)
            self.assertEqual(result['Task2']['arms'][ARMS[0]]['correct'],3)
            tampered=args.out/ARMS[1]/'Task2/predictions.csv'
            tampered.write_text(tampered.read_text().replace('0,1','0,0'))
            with patch(scorer,side_effect=AssertionError('Opened targets before verifying all four hashes')):
                with self.assertRaises(ValueError):
                    score(args,release)
