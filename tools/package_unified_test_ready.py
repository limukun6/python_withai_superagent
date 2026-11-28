"""Bundle both frozen, independently reloaded predictors and exercise the shipped copies offline."""
import argparse
import csv
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
from superagent.external_assets import activate_resource_directory
from tools.export_model_bundle import export
from tools.package_verified_delivery import LAUNCH, copy_resources


def package(out, notices):
    freeze = read(out / 'freeze.json')
    scores = read(out / 'validation.json')
    require(source_identity(ROOT) == freeze['solver'], 'Check out the frozen solver before packaging')
    activate_resource_directory(out / 'weights')
    archive = out / 'Unified-Task1-Task2-Test-Ready.zip'
    require(not archive.exists(), 'Use a fresh delivery filename')
    with tempfile.TemporaryDirectory(dir=out) as temp:
        work = Path(temp)
        tree = work / 'delivery'
        tree.mkdir()
        rows = []
        for key, name in [('a', 'Task1'), ('b', 'Task2')]:
            seal = freeze['exports'][key]
            folder = tree / name
            folder.mkdir()
            require(seal['independent_reload'] and not seal['course_test_opened']
                    and sha(out / key / 'reload.csv') == seal['predictions'], 'Missing frozen validation replay')
            bundle = export(out / key / 'run/agent', work / ('export_' + key))
            solution = folder / 'solution'
            shutil.move(bundle, solution)
            manifest = read(solution / 'bundle_manifest.json')
            require(manifest['model_sha256'] == seal['model'], 'Fitted model identity changed')
            shutil.copyfile(ROOT / 'unified_agent.py', solution / 'src/unified_agent.py')
            require(source_identity(solution / 'src') == freeze['solver'], 'Bundle solver differs')
            manifest['files']['src/unified_agent.py'] = sha(solution / 'src/unified_agent.py')
            (solution / 'bundle_manifest.json').write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
            copy_resources(manifest['external_resources'], folder, notices)
            shutil.copyfile(solution / 'requirements.txt', folder / 'requirements.txt')
            (folder / 'predict.py').write_text(LAUNCH, encoding='utf-8')
            resources = [r['model'] for r in manifest['external_resources']]
            save(folder / 'frozen_run.json', dict(evidence=seal, designated_validation=scores[key],
                                                  external_models=resources))
            rows.append(f"| {name} | {scores[key]['correct']} / {scores[key]['rows']} | "
                        f"{scores[key]['accuracy']:.2%} | {seal['fits']} |")
        (tree / 'README.md').write_text(
            '# Task1 和 Task2 冻结预测包\n\n两个模型从各自的原始训练数据自主选出。'
            '下表为课程指定验证集成绩。正式测试集待评。\n\n'
            '| 冻结模型 | 验证答对 | 准确率 | 实际拟合次数 |\n| --- | ---: | ---: | ---: |\n'
            + '\n'.join(rows) + '\n\n进入对应的 Task1 或 Task2 目录，用 Python 3.11 安装依赖后，'
            '传入该任务的无标签特征包。在 NVIDIA 机器上可以安装 requirements.txt 所列相同版本的 CUDA 版 PyTorch。'
            '训练选中的外部权重位于 assets，预测阶段可以断网。\n\n'
            '```sh\npython -m pip install -r requirements.txt\n'
            'python -I predict.py --features /path/to/features.zip --output predictions.csv\n```\n\n'
            'frozen_run.json 记录模型指纹、决策及验证结果。两个 solution/src 目录包含同一份训练源码。'
            '换机器时需先安装兼容的依赖和驱动，程序随后探测可用设备。\n',
            encoding='utf-8')
        stage = work / archive.name
        with zipfile.ZipFile(stage, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zipped:
            for file in sorted(tree.rglob('*')):
                if file.is_file():
                    zipped.write(file, file.relative_to(tree))
        relocated = work / 'relocated'
        with zipfile.ZipFile(stage) as zipped:
            zipped.extractall(relocated)
        cwd = work / 'receiver'
        cwd.mkdir()
        # No original project paths, weight locations or credentials in the receiver process.
        env = {k: os.environ[k] for k in ('PATH','SYSTEMROOT','WINDIR','TMPDIR','TEMP','TMP') if k in os.environ}
        env.update(OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        verified = {}
        for key, name in [('a', 'Task1'), ('b', 'Task2')]:
            output = cwd / (name + '.csv')
            subprocess.run([sys.executable, '-I', str(relocated / name / 'predict.py'),
                            '--features', str(out / key / 'queries.zip'), '--output', str(output)],
                           cwd=cwd, env=env, stdin=subprocess.DEVNULL, check=True, timeout=600)
            digest = sha(output)
            with output.open(encoding='utf-8-sig') as stream:
                count = sum(1 for _ in csv.DictReader(stream))
            seal = freeze['exports'][key]
            require(digest == seal['predictions'] and count == seal['rows'], 'Relocated output differs: ' + name)
            verified[name] = dict(rows=count, predictions=digest, selected_model=seal['model'])
        summary = dict(archive=archive.name, bytes=stage.stat().st_size, archive_sha256=sha(stage),
                       solver=freeze['solver'], verified=verified, model_api=0, new_fits=0, official_test_opened=False,
                       evidence='Extracted archive, independent isolated interpreter, offline, exact row and file match')
        stage.rename(archive)
        save(out / 'portable_receipt.json', summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=lambda x: Path(x).resolve())
    parser.add_argument('--notices', required=True, type=lambda x: Path(x).resolve())
    args = parser.parse_args()
    package(args.out, args.notices)
