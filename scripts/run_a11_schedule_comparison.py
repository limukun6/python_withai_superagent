"""Official CLI paired fixed/dynamic study, reusing the sealed A10 data and delivery driver."""
import argparse
from pathlib import Path
from scripts.run_a10_real_task_pilot import run

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--validate-only',action='store_true')
    run(parser.parse_args(),comparison=True)
