"""Study-only observation of members read from the supplied raw archive."""
import json
import os
from pathlib import Path
import runpy
import zipfile

runpy.run_path(str(Path(__file__).resolve().parents[1] / "ab_acceptance_guard" / "sitecustomize.py"))
if os.environ.get("UNIFIED_RAW_INPUT"):
    source = Path(os.environ["UNIFIED_RAW_INPUT"]).resolve()
    allowed = set(json.loads(os.environ["UNIFIED_RAW_MEMBERS"]))
    destination = Path(os.environ["UNIFIED_RAW_LOG"])
    original = zipfile.ZipFile.open

    def record(self, name, *args, **kwargs):
        if isinstance(self.filename, (str, os.PathLike)) and Path(self.filename).resolve() == source:
            member = name.filename if isinstance(name, zipfile.ZipInfo) else name
            with destination.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(member) + "\n")
            if member not in allowed:
                raise PermissionError("Member is outside this training protocol")
        return original(self, name, *args, **kwargs)

    zipfile.ZipFile.open = record
