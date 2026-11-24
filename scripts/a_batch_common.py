"""Small offline experiment journal; never imports credentials or calls services."""
import hashlib
import json
from pathlib import Path
import time


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((Path(root)/'superagent').rglob('*.py'))}


class Journal:
    def __init__(self, output, source, protocol):
        self.out=Path(output);self.out.mkdir(parents=True,exist_ok=False)
        self.source=Path(source);self.before=hashes(self.source);self.start=time.monotonic()
        save(self.out/'protocol.json', {**protocol,'source_files':self.before,'real_api_requests':0})

    def finish(self, results, **extra):
        save(self.out/'results.json', results)
        unchanged=self.before==hashes(self.source)
        save(self.out/'completion.json', {'status':'completed' if unchanged else 'source_changed',
            'source_unchanged':unchanged,'elapsed_seconds':time.monotonic()-self.start,
            'real_api_requests':0, **extra})
