"""Fetch public encoder files at pinned revisions and verify every byte before installation."""
import argparse
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile

from superagent.ab_sentence import ALTERNATIVES
from superagent.paired_encoder import VARIANTS
from tools.encoder_assets import MODELS, matches

DOWNLOADS = dict(MODELS)
for variant in VARIANTS:
    spec = ALTERNATIVES[variant]
    DOWNLOADS[spec["model"].split("/")[-1]] = dict(repo=spec["model"], revision=spec["revision"], files=spec["files"])


def fetch(weights, name, endpoint):
    spec = DOWNLOADS[name]
    folder = weights / name
    folder.mkdir(parents=True, exist_ok=True)
    for filename, identity in spec["files"].items():
        target = folder / filename
        if matches(target, identity):
            continue
        url = f"{endpoint}/{spec['repo']}/resolve/{spec['revision']}/{filename}"
        with NamedTemporaryFile(dir=folder, prefix=".downloading-", delete=False) as stream:
            temp = Path(stream.name)
        try:
            subprocess.run(["curl", "--fail", "--silent", "--show-error", "--location", "--retry", "1",
                            "--connect-timeout", "12", "--max-time", "1400", "--output", str(temp), url], check=True)
            if not matches(temp, identity):
                raise ValueError(f"Downloaded {name}/{filename} differs from its pinned bytes")
            temp.replace(target)
            print(name, filename, identity[0], "verified", flush=True)
        finally:
            temp.unlink(missing_ok=True)
    if not all(matches(folder/file, identity) for file, identity in spec["files"].items()):
        raise ValueError(f"Incomplete {name} installation")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--model", required=True, choices=tuple(DOWNLOADS))
    p.add_argument("--endpoint", choices=("https://huggingface.co", "https://hf-mirror.com"),
                   default="https://huggingface.co")
    args = p.parse_args()
    fetch(args.weights, args.model, args.endpoint)
