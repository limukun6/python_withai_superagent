"""Zero-fit reproduction of inherited stdin versus a live EOF watchdog on Windows."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from scripts.run_sa5_semantic_panel import save,sha

CHILD=r'''import faulthandler,os,subprocess,sys,threading,time
assert os.read(sys.stdin.fileno(),1)==b'G'
ready=threading.Event()
def watch():
    ready.set();os.read(sys.stdin.fileno(),1)
threading.Thread(target=watch,daemon=True).start();ready.wait();time.sleep(.2)
faulthandler.dump_traceback_later(3,repeat=False)
kw={'stdin':subprocess.DEVNULL} if sys.argv[1]=='explicit_devnull' else {}
r=subprocess.run([sys.executable,'-c','print("worker_started")'],capture_output=True,text=True,timeout=6,**kw)
print(r.stdout.strip(),flush=True)
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    save(a.output/'protocol.json',dict(script_sha256=sha(Path(__file__)),actual_classifier_fits=0,campus_API_requests=0,
        maximum_seconds_per_probe=9,purpose='minimal subprocess plumbing, no task or model training'))
    rows=[]
    for mode in ('inherit','explicit_devnull'):
        start=time.monotonic()
        with (a.output/(mode+'.log')).open('w',encoding='utf-8') as log:
            child=subprocess.Popen([sys.executable,'-c',CHILD,mode],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT)
            try:
                child.stdin.write(b'G');child.stdin.flush()
                try:code=child.wait(timeout=9);status='exited'
                except subprocess.TimeoutExpired:child.kill();code=child.wait();status='timeout'
            finally:child.stdin.close()
        text=(a.output/(mode+'.log')).read_text(encoding='utf-8')
        rows.append(dict(mode=mode,status=status,code=code,worker_started='worker_started' in text.splitlines(),elapsed_seconds=time.monotonic()-start))
    result=dict(status='completed',cases=rows,actual_classifier_fits=0,campus_API_requests=0)
    save(a.output/'results.json',result);save(a.output/'completion.json',result);print(json.dumps(result))


if __name__=='__main__':main()
