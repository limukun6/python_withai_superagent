import unittest
from unittest.mock import patch
from types import SimpleNamespace
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

    def test_cuda_that_advertises_but_cannot_execute_is_ineligible(self):
        from unittest.mock import Mock
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1)
        backend = SimpleNamespace(cuda=cuda, backends=SimpleNamespace(), float32=object(),
                                  arange=Mock(side_effect=RuntimeError('no kernel image is available')))
        with patch.dict('sys.modules', torch=backend):
            probes = cr.accelerator_probe()
        self.assertFalse(probes[0]['usable'])
        self.assertEqual(probes[0]['device'],'cuda:0')
        self.assertEqual(cr.device_for(0,profile={'accelerators':probes}),'cpu')

    def test_selects_roomiest_working_device_and_sizes_the_default_search(self):
        profile = {'accelerators':[dict(device='cuda:0',usable=True,free_bytes=128),
                                   dict(device='cuda:1',usable=True,free_bytes=512)]}
        self.assertEqual(cr.device_for(120,profile=profile),'cuda:1')
        for cpu, gib, ceiling in [(2,12,12),(8,1,12),(4,8,24),(8,3,24),(8,16,64)]:
            with self.subTest(cpu=cpu,gib=gib), patch.object(cr,'cpu_threads',return_value=cpu), \
                    patch.object(cr,'available_memory',return_value=gib*1024*cr.MIB):
                self.assertEqual(cr.default_fit_ceiling(),ceiling)

    def test_wall_time_estimate_leaves_expensive_confirmations_and_refits_affordable(self):
        rows = [dict(status='completed',candidate='method',training_seconds=t) for t in (20.,125.,170.)]
        cap = {'method':dict(classifier_fits=1)}
        self.assertEqual(cr.affordable_fits(40,1500,rows[:2],cap)[0],40)
        self.assertEqual(cr.affordable_fits(40,1500,rows,cap)[0],7)
        self.assertEqual(cr.affordable_fits(5,1500,rows,cap)[0],5)
        self.assertEqual(cr.affordable_fits(40,15,rows,cap)[0],1)
        self.assertEqual(cr.affordable_fits(0,15,rows,cap)[0],0)

    def test_time_protection_selects_the_highest_ranked_affordable_refit(self):
        caps={n:dict(classifier_fits=1,refit_classifier_fits=1) for n in ('fast','slow')}
        history=[dict(status='completed',candidate=n,training_seconds=t) for n,t in [('fast',5.),('slow',120.)]]
        exploration=dict(action='explore',members=['slow'])
        menu=[exploration,dict(action='stop',members=['slow']),dict(action='stop',members=['fast'])]
        chosen=cr.protect_refit(exploration,menu,history,caps,dict(seconds_left=80.,estimated_seconds_per_fit=30.))
        self.assertEqual((chosen['action'],chosen['members']),('stop',['fast']))
        self.assertIs(cr.protect_refit(exploration,menu,history,caps,dict(seconds_left=1000.,estimated_seconds_per_fit=30.)),exploration)
