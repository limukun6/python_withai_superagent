"""Match frozen experiment source hashes to commits reachable from this checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from tools.pretrained_study import read, write


def git(*args):
    return subprocess.check_output(['git', *args], cwd=Path(__file__).resolve().parents[1])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sources = {}
    stages = {}
    for file in sorted(args.root.glob('*/results.json')):
        protocol = read(file.parent / 'protocol.json')
        if 'locations' not in protocol:
            continue
        snapshot = protocol['source']
        key = hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        sources[key] = snapshot
        stages[file.parent.name] = key
    paths = set().union(*(snapshot.keys() for snapshot in sources.values()))
    missing = set(sources)
    commits = {}
    blob_hashes = {}
    for revision in git('rev-list', '--reverse', '--topo-order', 'HEAD').decode().splitlines():
        if not missing:
            break
        tree = git('ls-tree', '-rz', revision, '--', 'superagent', 'tools')
        blobs = {}
        for entry in tree.split(b'\0'):
            if not entry:
                continue
            header, path = entry.split(b'\t', 1)
            path = path.decode()
            if path in paths or (path.startswith('superagent/') and path.count('/') == 1 and path.endswith('.py')):
                blobs[path] = header.split()[2].decode()
        modules = {p for p in blobs if p.startswith('superagent/')}
        for key in list(missing):
            snapshot = sources[key]
            if modules != {p for p in snapshot if p.startswith('superagent/')} or not snapshot.keys() <= blobs.keys():
                continue
            for path in snapshot:
                blob = blobs[path]
                if blob not in blob_hashes:
                    blob_hashes[blob] = hashlib.sha256(git('cat-file', 'blob', blob)).hexdigest()
                if blob_hashes[blob] != snapshot[path]:
                    break
            else:
                commits[key] = revision
                missing.remove(key)
    if missing:
        raise ValueError('Frozen sources have no exact match in repository history: ' +
                         ', '.join(n for n,k in stages.items() if k in missing))
    write(args.output, dict(stages={n:dict(source_set_sha256=k,git_commit=commits[k]) for n,k in stages.items()},
                            source_sets=len(sources),scope='Exact runtime files and study driver for each stage. Reused controls keep their original sources.'))
    print(len(stages), 'stages matched to', len(sources), 'distinct source snapshots')


if __name__ == '__main__':
    main()
