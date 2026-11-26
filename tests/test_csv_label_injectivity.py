import csv
from pathlib import Path
import tempfile
import unittest

from scripts.sa5_semantic_cases import output
from superagent.output_recipe import write_export


class CSVLabelInjectivityTests(unittest.TestCase):
    def spec(self,mapping):
        recipe=output([dict(name='Prediction',kind='label',mapping=mapping)],delimiter=',',encoding='utf-8')
        return dict(metric_spec=dict(name='accuracy',labels=['A','B'],direction='maximize'),output_recipe=recipe)

    def test_distinct_json_scalars_that_collapse_in_csv_are_refused_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            for i,mapping in enumerate([{'A':0,'B':'0'},{'A':True,'B':'True'},{'A':1.,'B':'1.0'}]):
                dest=Path(temporary)/(str(i)+'.csv')
                with self.assertRaisesRegex(ValueError,'collapses'):
                    write_export(None,self.spec(mapping),dest,['A','B'])
                self.assertFalse(dest.exists())

    def test_scalar_types_can_differ_if_the_serialized_class_values_are_distinct(self):
        with tempfile.TemporaryDirectory() as temporary:
            dest=Path(temporary)/'predictions.csv'
            write_export(None,self.spec({'A':0,'B':'1'}),dest,['A','B'])
            with dest.open(encoding='utf-8',newline='') as stream:rows=list(csv.reader(stream))
            self.assertEqual(rows,[['Prediction'],['0'],['1']])


if __name__=='__main__':unittest.main()
