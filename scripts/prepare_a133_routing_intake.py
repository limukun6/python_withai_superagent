"""Extract a pinned B optional routing release safely, keeping exact Git blob bytes."""
import argparse
import io
from pathlib import Path,PurePosixPath
import subprocess
import tarfile
import time
from scripts.a_batch_common import save,hashes
from scripts.run_a10_real_task_pilot import sha

ROOT=Path(__file__).resolve().parents[1]
PROVIDER='924e88ab5f6b49d90d1baf2b835f533e644a5776'
PREFIX='releases/alternative-encoders-2026-11-23'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    save(a.output/'protocol.json',dict(batch='A133-intake',provider=PROVIDER,source_files=hashes(ROOT),
        runner_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'docs/batches/A133_A135_next_integration_and_task1_plan_20261123_zh.md'),
        maximum_classifier_fits=0,real_API_requests=0))
    try:
        raw=subprocess.check_output(['git','archive','--format=tar',PROVIDER,PREFIX],cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            for member in archive:
                name=PurePosixPath(member.name)
                if name.is_absolute() or '..' in name.parts or '\\' in member.name or not (member.isfile() or member.isdir()):
                    raise ValueError('Unsafe archive entry')
                target=a.output.joinpath(*name.parts)
                if not target.resolve().is_relative_to(a.output):raise ValueError('Archive path escapes output')
                if member.isdir():target.mkdir(parents=True,exist_ok=True)
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(archive.extractfile(member).read())
        package=a.output/PREFIX;patch=package/'docs/integration/A117_B_confidence_cascade.patch'
        save(a.output/'results.json',dict(package=str(package),patch=str(patch),patch_sha256=sha(patch),
            changed_paths=[line.split(' b/',1)[1] for line in patch.read_text(encoding='utf-8').splitlines() if line.startswith('diff --git')]))
        save(a.output/'completion.json',dict(status='extracted_pending_review',actual_classifier_fits=0,real_API_requests=0,
            elapsed_seconds=time.monotonic()-start))
        print(dict(package=str(package),patch_sha256=sha(patch)),flush=True)
    except Exception as error:
        save(a.output/'completion.json',dict(status='stopped',error=str(error),actual_classifier_fits=0,real_API_requests=0));raise


if __name__=='__main__':main()
