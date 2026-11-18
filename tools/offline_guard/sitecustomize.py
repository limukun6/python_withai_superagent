import os
import sys

if os.environ.get('SUPERAGENT_OFFLINE')=='1':
    def offline(event,args):
        if event=='socket.connect' or event=='open' and isinstance(args[0],str) and '/api-test/' in args[0]:
            raise RuntimeError('This experiment cannot access the network or credentials')
    sys.addaudithook(offline)
