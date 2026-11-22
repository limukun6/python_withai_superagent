"""A74: reject network in every child, observe worker classifier fits."""
import os
from pathlib import Path
import runpy
import sys

if os.environ.get('A74_OFFLINE') == '1':
    def offline(event, args):
        if event in ('socket.connect', 'socket.getaddrinfo', 'socket.sendto'):
            raise PermissionError('A74 network forbidden')
    sys.addaudithook(offline)
    runpy.run_path(str(Path(os.environ['A74_SOURCE_ROOT'])/'scripts/a29_worker_observer/sitecustomize.py'))
