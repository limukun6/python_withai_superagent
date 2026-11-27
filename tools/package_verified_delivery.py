"""Make a portable, offline copy of an independently verified validation delivery."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

from scripts.run_planner_self_review import source_identity
from scripts.unified_released_study import ROOT, read, require, save, sha
from tools.export_model_bundle import export


LAUNCH = '''"""Run the bundled frozen predictor using the bundled encoder files."""
import json
import os
from pathlib import Path
import sys

root = Path(__file__).resolve().parent
solution = root / 'solution'
manifest = json.loads((solution / 'bundle_manifest.json').read_text(encoding='utf-8'))
env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', SUPERAGENT_OFFLINE='1',
           TOKENIZERS_PARALLELISM='false')
for resource in manifest['external_resources']:
    path = root / 'assets' / resource['name']
    if resource['path_kind'] == 'file':
        path = path / 'weights'
    env[resource['environment_variable']] = str(path)
os.execve(sys.executable, [sys.executable, '-I', str(solution / 'predict.py'), *sys.argv[1:]], env)
'''


def copy_resources(resources, destination, notices):
    for resource in resources:
        name = resource['name']
        require(Path(name).name == name and name not in ('.', '..'), 'Unexpected resource name')
        source = Path(os.environ[resource['environment_variable']])
        folder = destination / 'assets' / name
        folder.mkdir(parents=True)
        if resource['path_kind'] == 'file':
            entries = [(source, 'weights', resource['file'])]
        else:
            entries = [(source / rel, rel, spec) for rel, spec in resource['files'].items()]
        for path, rel, spec in entries:
            target = folder / rel
            require(target.resolve().is_relative_to(folder.resolve()), 'Resource file outside its folder')
            target.parent.mkdir(exist_ok=True, parents=True)
            shutil.copyfile(path, target)
            require(target.stat().st_size == spec['bytes'] and sha(target) == spec['sha256'],
                    'Copied encoder file differs from the pinned resource')
        source_notice = notices / name
        identity = read(source_notice / 'source.json')
        require(identity['model'] == resource['model'] and identity.get('revision') == resource.get('revision'),
                'Model attribution does not match the fitted encoder')
        notice_target = destination / 'third_party' / name
        notice_target.mkdir(parents=True)
        for filename in ('source.json', 'LICENSE.txt', 'model-card.md', 'NOTICE.txt'):
            path = source_notice / filename
            if filename != 'NOTICE.txt' or path.is_file():
                shutil.copyfile(path, notice_target / filename)


def package(args):
    receipt = read(args.run / 'validation.json')
    require(receipt['runtime_sha256'] == source_identity(ROOT) and receipt['independent_reload'],
            'Use the frozen training source and the scored, reloaded validation delivery')
    input_spec = read(args.run / 'protocol.json')['packages']['titles']
    features = args.run / 'titles/features.zip'
    original = args.run / 'titles/delivery/predictions.csv'
    require(sha(features) == input_spec['features'] and sha(original) == receipt['prediction_sha256'],
            'Registered validation input or predictions changed')
    archive_name = 'Released-Task2-validation-' + receipt['runtime_sha256'][:12] + '.zip'
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / archive_name
    require(not archive.exists() and not (args.output / 'portable_receipt.json').exists(),
            'Use a fresh delivery destination')
    with tempfile.TemporaryDirectory(dir=args.output) as scratch:
        scratch = Path(scratch)
        tree = scratch / 'delivery'
        tree.mkdir()
        solution = tree / 'solution'
        bundle = export(args.run / 'titles/run/agent', scratch / 'export')
        shutil.move(bundle, solution)
        manifest = read(solution / 'bundle_manifest.json')
        require(manifest['model_sha256'] == receipt['model_sha256'], 'Model differs from scored run')
        copy_resources(manifest['external_resources'], tree, args.notices)
        shutil.copyfile(ROOT / 'unified_agent.py', solution / 'src/unified_agent.py')
        require(source_identity(solution / 'src') == receipt['runtime_sha256'], 'Unified source copy differs')
        shutil.copyfile(ROOT / 'requirements.txt', solution / 'src/requirements.txt')
        shutil.copyfile(solution / 'requirements.txt', tree / 'requirements.txt')
        (tree / 'predict.py').write_text(LAUNCH, encoding='utf-8')
        results = tree / 'results'
        results.mkdir()
        for src, dst in [(original, 'validation_predictions.csv'), (args.run / 'validation.json', 'validation.json'),
                         (args.run / 'protocol.json', 'run_protocol.json')]:
            shutil.copyfile(src, results / dst)
        logs = results / 'run_logs'
        logs.mkdir()
        agent = args.run / 'titles/run/agent'
        for name in ('task_contract.json', 'run_manifest.json', 'decisions.json', 'history.json',
                     'classifier_fits.json', 'events.jsonl', 'recovery_state.json', 'run_outcome.json', 'llm_usage.jsonl'):
            shutil.copyfile(agent / name, logs / name)
        for name in ('analogy_review.json','analogy_incumbent.json','analogy_selection.json'):
            if (agent/name).is_file():
                shutil.copyfile(agent/name,logs/name)
        shutil.copyfile(args.run / 'titles/physical.jsonl', logs / 'observed_classifier_fits.jsonl')
        (tree / 'README.md').write_text(
            '# Released Task2 离线交接包\n\n'
            '本包对应一次从课程训练原包开始的自主运行。指定验证集逐行输出和分数在 results 目录。'
            'run_logs 保存决策、训练、计费和交付记录。\n\n'
            '环境为 CPython 3.11。用同一环境依次运行下面两行，把 features.zip 换成待预测的特征包。'
            '依赖安装好后，预测时可断网，包内已经带有所选模型的权重。\n\n'
            '```sh\npython -m pip install -r requirements.txt\n'
            'python -I predict.py --features features.zip --output predictions.csv\n```\n\n'
            'solution/src 是这次训练使用的统一入口和生产源码。solution 中保存已训练的模型、'
            '选择记录和逐文件资源清单。third_party 收录外部模型的固定版本、作者说明和许可证。\n',
            encoding='utf-8')
        save(tree / 'files.json', {path.relative_to(tree).as_posix(): sha(path)
             for path in sorted(tree.rglob('*')) if path.is_file()})
        staged = Path(shutil.make_archive(str(scratch / 'package'), 'zip', root_dir=tree))
        relocated = scratch / 'independent-reload'
        with zipfile.ZipFile(staged) as zipped:
            zipped.extractall(relocated)
        for rel, expected in read(relocated / 'files.json').items():
            require(sha(relocated / rel) == expected, 'Relocated file differs: ' + rel)
        result = scratch / 'check/predictions.csv'
        # The child starts with no project paths, resource variables or credentials.
        environment = {k: os.environ[k] for k in ('PATH', 'SYSTEMROOT', 'WINDIR', 'TMPDIR', 'TEMP', 'TMP')
                       if k in os.environ}
        environment.update(OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        subprocess.run([sys.executable, '-I', str(relocated / 'predict.py'), '--features', str(features),
                        '--output', str(result)], cwd=relocated, env=environment,
                       stdin=subprocess.DEVNULL, check=True, timeout=600)
        require(sha(result) == receipt['prediction_sha256'], 'Portable replay differs from original delivery')
        summary = dict(archive=archive_name, bytes=staged.stat().st_size, archive_sha256=sha(staged),
                       runtime_sha256=receipt['runtime_sha256'], model_sha256=receipt['model_sha256'],
                       predictions_sha256=sha(result), rows=receipt['rows'], independent_relocated_replay=True,
                       installed_python=sys.version.split()[0], resource_models=[
                           dict(name=r['model'], revision=r.get('revision')) for r in manifest['external_resources']],
                       network_forbidden_for_prediction=True, new_classifier_fits=0, new_model_requests=0)
        staged.rename(archive)
        save(args.output / 'portable_receipt.json', summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('run', 'notices', 'output'):
        parser.add_argument('--' + arg, required=True, type=lambda value: Path(value).resolve())
    package(parser.parse_args())
