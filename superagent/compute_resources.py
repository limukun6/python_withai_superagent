"""Discover usable local compute. Inference retries never spend another supervised fit."""
from functools import lru_cache
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

MIB = 2**20
_WORKER_PROFILE = 'SUPERAGENT_PARENT_COMPUTE'
_events = {}
_disabled = set()


def cpu_threads(ceiling=8):
    from joblib import cpu_count
    return max(1, min(ceiling, cpu_count()))


def available_memory():
    try:
        import psutil
        available = int(psutil.virtual_memory().available)
    except (ImportError, OSError):
        try:
            available = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE')
        except (OSError, ValueError, AttributeError):
            available = 512*MIB
    for limit, used in [('memory.max','memory.current'),('memory/memory.limit_in_bytes','memory/memory.usage_in_bytes')]:
        try:
            cgroup = Path('/sys/fs/cgroup')
            available = min(available, max(0,int((cgroup/limit).read_text())-int((cgroup/used).read_text())))
        except (OSError, ValueError):
            pass
    return max(0, available)


def accelerator_probe():
    import torch
    devices, observations = [], []
    if torch.cuda.is_available():
        devices.extend('cuda:'+str(i) for i in range(torch.cuda.device_count()))
    if getattr(getattr(torch.backends,'mps',None),'is_available',lambda:False)():
        devices.append('mps')
    for name in devices:
        try:
            v = torch.arange(16, device=name, dtype=torch.float32).reshape(4,4)
            result = (v @ v.T).sum().item()
            if result != 3680.0:
                raise RuntimeError('Incorrect accelerated matrix product')
            if name.startswith('cuda'):
                torch.cuda.synchronize(name)
                free, _ = torch.cuda.mem_get_info(name)
            else:
                torch.mps.synchronize()
                free = min(available_memory(), max(0, torch.mps.recommended_max_memory()-torch.mps.driver_allocated_memory()))
            observations.append(dict(device=name, usable=True, free_bytes=int(free), verified='matrix product and host readback'))
        except (RuntimeError, NotImplementedError, AssertionError, OSError, AttributeError) as exc:
            observations.append(dict(device=name, usable=False, reason=str(exc)[:240]))
    return observations


@lru_cache(maxsize=1)
def inventory():
    from importlib.util import find_spec
    # Only a worker explicitly spawned by our controller inherits this live snapshot.
    inherited = os.environ.get(_WORKER_PROFILE)
    if inherited and 'superagent.experiment_worker' in getattr(sys,'orig_argv',()):
        return json.loads(inherited)
    info = dict(system=platform.system(), architecture=platform.machine(), cpu_threads=cpu_threads(),
                available_memory_bytes=available_memory(), accelerators=[], probe='torch unavailable')
    if find_spec('torch') is not None:
        try:
            child = subprocess.run([sys.executable,'-m','superagent.compute_resources'], stdin=subprocess.DEVNULL,
                                   cwd=Path(__file__).resolve().parents[1],capture_output=True,text=True,timeout=18)
            if child.returncode == 0:
                info.update(accelerators=json.loads(child.stdout), probe='completed')
            else:
                info['probe']='accelerator probe failed in isolated process (exit '+str(child.returncode)+')'
        except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
            info['probe']='CPU fallback after '+type(exc).__name__
    return info


def default_fit_ceiling():
    cpu,ram=cpu_threads(),available_memory()
    if cpu<=2 or ram<2*1024*MIB:
        return 12
    if cpu<=4 or ram<4*1024*MIB:
        return 24
    return 64


def worker_environment():
    return dict(os.environ, **{_WORKER_PROFILE:json.dumps(inventory(),separators=(',',':'))})


def device_for(required_bytes, *, worthwhile=True, profile=None):
    data=inventory() if profile is None else profile
    eligible=[d for d in data['accelerators'] if worthwhile and d['usable'] and d['device'] not in _disabled
              and d['free_bytes']>=required_bytes]
    return max(eligible,key=lambda d:d['free_bytes'])['device'] if eligible else 'cpu'


def note(workload, device, batch=None, reason=None):
    key=(workload,device,reason)
    item=_events.setdefault(key,dict(workload=workload,device=device,batches=0))
    item['batches']+=1
    if batch is not None:
        item['max_batch']=max(batch,item.get('max_batch',0))
    if reason:
        item['reason']=reason


def decisions():
    return list(_events.values())


def device_error(error):
    message=str(error).lower()
    return isinstance(error,NotImplementedError) or any(t in message for t in
        ('out of memory','cuda','cublas','cudnn','mps','metal','accelerator','not implemented for'))


def disable(device,workload,error):
    _disabled.add(device)
    note(workload,device,reason=('switch to CPU: '+str(error))[:220])


def batched_forward(model, order, widths, maximum, evaluate, *, workload):
    """Keep order and retry only the unfinished frozen batch, shrinking it on memory pressure."""
    parameter_bytes=sum(p.numel()*p.element_size() for p in model.parameters())
    width=max(1,int(max(widths,default=1)))
    hidden=int(getattr(getattr(model,'config',None),'hidden_size',768))
    working_per_row=max(MIB,width*hidden*4*16 + width*width*4*12)
    needed=parameter_bytes*2+max(96*MIB,working_per_row*2)
    selected=device_for(needed,worthwhile=len(order)>=8 and parameter_bytes>=MIB)
    memory=inventory()['available_memory_bytes'] if selected=='cpu' else next(
        d['free_bytes'] for d in inventory()['accelerators'] if d['device']==selected)
    batch=min(maximum,max(1,(memory//2-parameter_bytes)//working_per_row))
    start=0
    while start<len(order):
        ids=order[start:start+batch]
        try:
            model.to(selected)
            out=evaluate(ids,selected)
        except (RuntimeError,NotImplementedError) as exc:
            if selected=='cpu' or not device_error(exc):
                raise
            if 'out of memory' in str(exc).lower() and batch>1:
                batch=max(1,batch//2)
                note(workload,selected,reason='reduce batch after memory pressure')
            else:
                disable(selected,workload,exc)
                selected='cpu'
            continue
        note(workload,selected,len(ids))
        yield ids,out
        start+=len(ids)


if __name__=='__main__':
    print(json.dumps(accelerator_probe()))
