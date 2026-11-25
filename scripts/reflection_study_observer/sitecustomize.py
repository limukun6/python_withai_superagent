"""Allow only the campus client in the designated controller; keep fit workers offline and counted."""
import os
from pathlib import Path
import runpy
import sys


if os.getenv('AB_B1_OFFLINE') == '1':
    paid = os.getenv('REFLECTION_STUDY_LIVE') == '1'
    args = sys.orig_argv
    private_evaluation=os.getenv('PROPOSAL_STUDY_OUTER_LABELS')
    feature_packet=os.getenv('PROPOSAL_STUDY_OUTER_FEATURES')
    if private_evaluation:
        training=('superagent' in args and 'unified' in args) or 'superagent.experiment_worker' in args
        def protect_outer(event, arguments):
            if event=='open' and isinstance(arguments[0],(str,bytes)):
                path=os.path.abspath(os.fsdecode(arguments[0]))
                if path.startswith(private_evaluation+os.sep) or (training and path==feature_packet):
                    raise PermissionError('Evaluation data is sealed from this process')
        sys.addaudithook(protect_outer)
    if paid and 'superagent' in args and 'unified' in args and '--feedback-reflection' in args:
        from scripts.campus_direct import CAMPUS_IP, campus_resolution
        os.environ['SUPERAGENT_OFFLINE'] = '0'
        binding = campus_resolution(CAMPUS_IP)
        binding.__enter__()

        def campus_only(event, arguments):
            if event in ('socket.connect','socket.sendto'):
                address = arguments[1] if event == 'socket.connect' else arguments[-1]
                if not isinstance(address,tuple) or address[:2] != (CAMPUS_IP,443):
                    raise PermissionError('Only the campus HTTPS connection is enabled')
        sys.addaudithook(campus_only)
    else:
        os.environ['SUPERAGENT_OFFLINE'] = '1'
        runpy.run_path(str(Path(__file__).resolve().parents[1]/'ab_acceptance_guard/sitecustomize.py'))
