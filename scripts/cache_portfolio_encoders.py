"""Cache only independent frozen features from the training portion of an anonymous preference task."""
import argparse
import os
from pathlib import Path

from scripts.unified_released_study import sha
from superagent.data import load_training_task
from tools.encoder_assets import MODELS, verified
from tools.portfolio_encoders import build_bank, keys_for


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train", type=Path, required=True)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--model", choices=tuple(MODELS), required=True)
    args = p.parse_args()
    task = load_training_task(args.train)
    if task.kind != "pair_text" or task.spec["input_schema"].get("pair_semantics") != "preference":
        raise ValueError("This portfolio requires declared pair preferences")
    os.environ["SUPERAGENT_PORTFOLIO_WEIGHTS"] = str(args.weights.resolve())
    verified(args.weights, args.model)
    print("Encoding", args.model, len(set(keys_for(args.model, task.X_train))), "unique feature inputs", flush=True)
    file = build_bank(args.model, task.X_train, args.cache)
    print("Saved", args.model, sha(file), flush=True)


if __name__ == "__main__":
    main()
