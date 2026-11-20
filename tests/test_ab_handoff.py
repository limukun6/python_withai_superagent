import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import joblib
import numpy as np
from superagent.prediction_evidence import make_prediction_record
from tools.export_b_handoff import export


class HandoffTests(unittest.TestCase):
    def test_prediction_order_and_budget_are_preserved_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def save(name,value):(root/name).write_text(json.dumps(value),encoding='utf-8')
            save('task_contract.json',{'kind':'numeric_tabular','metric':{'name':'accuracy'}})
            save('split.json',{'search_indices':[7,2,9]})
            save('trials.json',[{'trial':1,'candidate':'one','status':'completed','classifier_fits':1}])
            save('decision_states.json',[{'available_candidates':[{'id':'one','params':{}},{'id':'two','params':{}}]}])
            (root/'llm_usage.jsonl').write_text(json.dumps({'event':'ledger_initialized'})+'\n',encoding='utf-8')
            payload=root/'inner_data.joblib';truth=np.array([0,1,1]);pred=np.array([0,0,1])
            joblib.dump({'y_search':truth},payload)
            record=make_prediction_record(pred,truth,hashlib.sha256(payload.read_bytes()).hexdigest())
            artifact=root/'trial_1.joblib';joblib.dump({'prediction_record':record},artifact)
            result=export(root)
            self.assertEqual(result['prediction_evidence'][0]['row_ids'],[7,2,9])
            self.assertEqual(result['prediction_evidence'][0]['predictions'],[0,0,1])
            self.assertEqual(result['available_candidates'][0]['id'],'two')
            self.assertEqual(result['budget']['remaining_fit_allowance'],10)
            self.assertEqual(result['budget']['reserved_final_refit'],1)
            record['predictions']=np.array([1,0,1]);joblib.dump({'prediction_record':record},artifact)
            with self.assertRaisesRegex(ValueError,'Invalid prediction'):export(root)

if __name__=='__main__':unittest.main()
