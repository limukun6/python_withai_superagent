"""Attach one training-feedback decision to an existing unified run, or predict with the frozen choice."""
import argparse
from pathlib import Path

from superagent.predict import create_predictions
from tools.unified_handoff import attach, frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    attach_cmd = commands.add_parser("attach")
    predict_cmd = commands.add_parser("predict")
    for cmd in (attach_cmd, predict_cmd):
        cmd.add_argument("--run-dir", type=Path, required=True)
    predict_cmd.add_argument("features", type=Path)
    predict_cmd.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "attach":
        result = attach(args.run_dir)
        print(result["chosen"], result["reason"], "extra_fits", result["extra_fits"])
    else:
        _, native = frozen(args.run_dir)
        print(create_predictions(str(args.features), native, args.output))


if __name__ == "__main__":
    main()
