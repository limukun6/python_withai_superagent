"""Explicit process-only campus DNS selection; HTTPS hostname and certificate checks stay intact."""
import argparse
from contextlib import contextmanager
import json
import runpy
import socket
import sys
from unittest.mock import patch


HOST='aigc-api.hkust-gz.edu.cn'
CAMPUS_IP='10.121.10.250'


@contextmanager
def campus_resolution(address):
    if address!=CAMPUS_IP:raise ValueError('Only the documented campus endpoint address is permitted')
    original=socket.getaddrinfo
    def resolve(host,port,*args,**kwargs):
        name=host.decode('ascii') if isinstance(host,bytes) else host
        target=address if isinstance(name,str) and name.rstrip('.').casefold()==HOST else host
        return original(target,port,*args,**kwargs)
    with patch.object(socket,'getaddrinfo',resolve):yield


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--ip',required=True,choices=[CAMPUS_IP]);p.add_argument('module',choices=['superagent'])
    p.add_argument('arguments',nargs=argparse.REMAINDER);args=p.parse_args()
    print(json.dumps(dict(explicit_process_dns=dict(host=HOST,address=args.ip),TLS_hostname_preserved=True,
        persistent_network_configuration_modified=False,automatic_retry=False)),flush=True)
    with campus_resolution(args.ip):
        sys.argv=[args.module,*args.arguments]
        runpy.run_module(args.module,run_name='__main__',alter_sys=True)
