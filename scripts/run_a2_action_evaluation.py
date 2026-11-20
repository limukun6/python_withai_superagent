"""Evaluate complete sealed A2 decisions; --fixture is explicitly non-LLM evidence."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.a2_action_evaluation import evaluate

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True);p.add_argument('--calls',required=True)
    p.add_argument('--output',required=True);p.add_argument('--fixture',action='store_true');p.add_argument('--validate-only',action='store_true')
    a=p.parse_args();evaluate(a.prepared,a.calls,a.output,a.fixture,a.validate_only)
