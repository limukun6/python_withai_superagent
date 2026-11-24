"""Read-only reuse of trusted local classifier artifacts for an identical task and fit."""
from importlib.metadata import version
import os
from pathlib import Path
import shutil

from scripts.unified_released_study import ROOT, read, require, sha
from unified_agent import save

FILES = ("trained.json", "admission.json", "fits.json", "model.joblib")
SOURCES = ("tools/portfolio_training.py", "tools/portfolio_encoders.py", "tools/encoder_assets.py",
           "superagent/ab_sentence.py", "superagent/word_familiarity.py", "superagent/ab_methods.py",
           "superagent/fit_guard.py", "superagent/models.py", "requirements-ab-encoder-tested.txt", "requirements-tested.txt")
LIBRARIES = ("scikit-learn", "numpy", "scipy", "joblib", "torch", "transformers", "tokenizers", "safetensors")


def recipe():
    return {"files": {n: sha(ROOT / n) for n in SOURCES}, "runtime": {n: version(n) for n in LIBRARIES}}


def copy_component(source, target, hashes):
    require(not target.exists() and set(hashes) == set(FILES), "Unexpected cache entry")
    require(all(sha(source / f) == hashes[f] for f in FILES), "Component differs from the trusted index")
    target.mkdir(parents=True)
    for f in FILES:
        try:
            os.link(source / f, target / f)
        except OSError:
            shutil.copyfile(source / f, target / f)


def build(cache, package, components):
    cache.mkdir(parents=True, exist_ok=False)
    entries = {}
    for folder in components:
        rec = read(folder / "trained.json")
        require(rec["identity"]["package"] == package, "Attempt to reuse a different training package")
        group, name = rec["identity"]["ids"], rec["name"]
        require(len(group) == 64 and group.isalnum() and name == folder.name, "Invalid component identity")
        view = entries.setdefault(group, {})
        # Multiple independent fits may produce different files. Keep the explicitly preferred first source.
        if name in view:
            continue
        hashes = {f: sha(folder / f) for f in FILES}
        require(rec["model"] == hashes["model.joblib"] and rec["ledger"] == hashes["fits.json"], "Unbound fit receipt")
        copy_component(folder, cache / group / name, hashes)
        view[name] = hashes
    save(cache / "index.json", {"package": package, "recipe": recipe(), "entries": entries})


class FitCache:
    def __init__(self, path, package):
        self.path = Path(path) if path else None
        self.entries = {}
        if self.path:
            index = read(self.path / "index.json")
            require(index["package"] == package and index["recipe"] == recipe(),
                    "Trusted fit cache belongs to another package, recipe or environment")
            self.entries = index["entries"]

    def restore(self, target, name, identity):
        group = identity["ids"]
        hashes = self.entries.get(group, {}).get(name)
        if hashes is None:
            return False
        source = self.path / group / name
        rec = read(source / "trained.json")
        ledger = read(source / "fits.json")
        require(rec["name"] == name and rec["identity"] == identity and
                read(source / "admission.json")["training_identity"] == identity and
                rec["model"] == hashes["model.joblib"] and rec["ledger"] == hashes["fits.json"] and
                ledger["attempted_classifier_fits"] == ledger["completed_classifier_fits"] == 1 and
                ledger["status"] == "completed" and rec["model_api_calls"] == 0, "Incompatible fit lineage")
        copy_component(source, target, hashes)
        return True
