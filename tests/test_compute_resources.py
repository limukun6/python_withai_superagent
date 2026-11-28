import unittest
from unittest.mock import patch
import numpy as np
from superagent import compute_resources as cr


class ComputeTests(unittest.TestCase):
    def test_failed_and_low_memory_are_ineligible(self):
        devices=[dict(device='cuda:0',usable=False,reason='no kernel image'),dict(device='cuda:1',usable=True,free_bytes=200)]
        self.assertEqual(cr.device_for(300,profile={'accelerators':devices}),'cpu')
        self.assertEqual(cr.device_for(100,profile={'accelerators':devices}),'cuda:1')
        self.assertEqual(cr.device_for(100,worthwhile=False,profile={'accelerators':devices}),'cpu')

    def test_oom_and_broken_kernel_replay_only_pending_batch(self):
        class Param:
            def numel(self):return 1024*1024
            def element_size(self):return 4
        class Net:
            def parameters(self):return [Param()]
            def to(self,device):return self
        data=dict(available_memory_bytes=3*1024**3,accelerators=[dict(device='cuda:1',usable=True,free_bytes=2*1024**3)])
        outputs=[]
        def forward(ids,device):
            if device!='cpu' and len(ids)>3:raise RuntimeError('CUDA out of memory')
            if device!='cpu' and ids[0]>=3:raise RuntimeError('CUDA no kernel image')
            outputs.append((device,ids.tolist()))
            return ids*2
        with patch.object(cr,'inventory',return_value=data), patch.object(cr,'_disabled',set()), patch.object(cr,'_events',{}):
            values=list(cr.batched_forward(Net(),np.arange(12),[16]*12,6,forward,workload='test'))
            self.assertEqual(np.concatenate([out for _,out in values]).tolist(),list(range(0,24,2)))
            self.assertEqual([name for name,_ in outputs],['cuda:1','cpu','cpu','cpu'])
            self.assertEqual(cr.device_for(1),'cpu')

    def test_threads_follow_available_cpu_quota(self):
        with patch('joblib.cpu_count',return_value=2):
            self.assertEqual(cr.cpu_threads(6),2)
        with patch('joblib.cpu_count',return_value=40):
            self.assertEqual(cr.cpu_threads(6),6)
