"""Package both raw-entry models, fixed local encoders and the exact source, then verify a clean extraction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

from scripts.unified_released_study import ROOT, read, require, save, sha
from superagent import ab_sentence

GUIDE = """# 两项课程任务的统一 SuperAgent

本包包含两次原始包自主运行选出的分类模型、必要的编码器、数据预处理和同版本入口。验证分数与训练数据量见 designated-results.json 和 models 下各自的 training_protocol.json。训练内留出的评分在标题模型的 audit.json。

使用 Python 3.11，安装 requirements-ab-encoder-tested.txt 中的依赖。在解压目录配置随包权重，可以离线预测或重新训练。

```bash
export SUPERAGENT_MINILM_WEIGHTS="$PWD/weights/all-MiniLM-L6-v2"
export SUPERAGENT_DEEP_MINILM_WEIGHTS="$PWD/weights/all-MiniLM-L12-v2"
export SUPERAGENT_E5_SMALL_WEIGHTS="$PWD/weights/e5-small-v2"
python -m unified_agent predict TABLE_FEATURES.zip --run-dir models/table --output table-output/predictions.csv
python -m unified_agent predict TITLE_FEATURES.zip --run-dir models/titles --output title-output/predictions.csv
python -m unified_agent train TASK.zip --run-dir runs/new-task
```

投递表格特征时，预测 ZIP 可采用课程的 C_test.npy。标题特征可采用 test.csv，使用训练时的 headline_A、headline_B 列。输入可以只含特征。训练命令接收课程原始包，各次训练使用新的运行目录。随包三个编码器的发布者、固定版本和文件指纹见 weights.json。运行环境依赖需要另行安装。
"""



def selected_artifact(agent, manifest):
    name = manifest.get("model_file", "selected_model.joblib")
    require(isinstance(name, str) and name not in ("", ".", "..") and
            Path(name).name == name and "/" not in name and chr(92) not in name, "Invalid frozen model filename")
    path = agent / name
    require(path.is_file() and (not manifest.get("model_sha256") or sha(path) == manifest["model_sha256"]),
            "Frozen selection and model file disagree")
    return path


def model_files(run):
    agent = run / "agent"
    files = [agent / "task_spec.json", run / "training_protocol.json"]
    for name in ("final_selection.json", "incumbent_selection.json"):
        manifest = agent / name
        if name == "final_selection.json" or manifest.is_file():
            files.extend([manifest, selected_artifact(agent, read(manifest))])
    if (agent / "stop_review.json").is_file():
        files.append(agent / "stop_review.json")
    return files


def packed_source(raw, weights):
    files = {p.relative_to(ROOT).as_posix(): p for p in (ROOT / "superagent").rglob("*.py")}
    for name in ("unified_agent.py", "requirements-tested.txt", "requirements-ab-encoder-tested.txt"):
        files[name] = ROOT / name
    metadata = {}
    for identifier, revision, recorded in (
        (ab_sentence.MODEL, ab_sentence.REVISION, ab_sentence.FILES),
        (ab_sentence.DEEP_MODEL, ab_sentence.DEEP_REVISION, ab_sentence.DEEP_FILES),
        (ab_sentence.ALTERNATIVES["e5"]["model"], ab_sentence.ALTERNATIVES["e5"]["revision"], ab_sentence.ALTERNATIVES["e5"]["files"]),
    ):
        folder = identifier.split("/")[-1]
        metadata[identifier] = {"revision": revision, "sha256": {name: values[1] for name, values in recorded.items()}}
        for name, (size, digest) in recorded.items():
            path = weights / folder / name
            require(path.stat().st_size == size and sha(path) == digest, "Missing exact encoder asset")
            files[f"weights/{folder}/{name}"] = path
    for tag in ("table", "titles"):
        run = raw / tag / "run"
        for artifact in model_files(run):
            files[f"models/{tag}/{artifact.relative_to(run).as_posix()}"] = artifact
        if (run / "audit.json").is_file():
            files[f"models/{tag}/audit.json"] = run / "audit.json"
    files["designated-results.json"] = raw / "results.json"
    return files, metadata


def package(raw, weights, output):
    files, metadata = packed_source(raw, weights)
    in_memory = {"weights.json": (json.dumps(metadata, indent=2) + "\n").encode(), "README.md": GUIDE.encode()}
    hashes = {name: sha(file) for name, file in files.items()}
    hashes.update({name: hashlib.sha256(data).hexdigest() for name, data in in_memory.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for name, file in files.items():
            z.write(file, name, compress_type=zipfile.ZIP_STORED if name.endswith(".safetensors") else zipfile.ZIP_DEFLATED)
        for name, data in in_memory.items():
            z.writestr(name, data)
        z.writestr("SHA256SUMS", "".join(hashes[name] + "  " + name + "\n" for name in sorted(hashes)))


def verify(raw, output):
    results = read(raw / "results.json")
    checked = {}
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "extracted"
        with zipfile.ZipFile(output) as z:
            z.extractall(root)
        expected = {s[66:]: s[:64] for s in (root / "SHA256SUMS").read_text().splitlines()}
        require({p.relative_to(root).as_posix(): sha(p) for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS"}
                == expected, "Extracted archive differs")
        ep = dict(os.environ, SUPERAGENT_OFFLINE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        for key, folder in (("MINILM", "all-MiniLM-L6-v2"), ("DEEP_MINILM", "all-MiniLM-L12-v2"), ("E5_SMALL", "e5-small-v2")):
            ep["SUPERAGENT_" + key + "_WEIGHTS"] = str(root / "weights" / folder)
        bootstrap = "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));sys.argv[0]='unified_agent';runpy.run_module('unified_agent',run_name='__main__')"
        for tag in ("table", "titles"):
            dest = Path(tmp) / tag / "predictions.csv"
            subprocess.run([sys.executable, "-I", "-B", "-c", bootstrap, str(root), "predict", str(raw / "designated" / (tag + ".zip")),
                            "--run-dir", str(root / "models" / tag), "--output", str(dest)], cwd=root, env=ep, check=True, timeout=300,
                           stdout=subprocess.DEVNULL)
            agent = root / f"models/{tag}/agent"
            model = sha(selected_artifact(agent, read(agent / "final_selection.json")))
            require(sha(dest) == results[tag]["prediction"] and model == results[tag]["model"], "Portable model changed predictions")
            checked[tag] = {"model": model, "prediction": sha(dest), "rows": results[tag]["rows"]}
    save(raw / "portable.json", {"archive_sha256": sha(output), "bytes": output.stat().st_size,
         "entrypoint": expected["unified_agent.py"], "models": checked, "fixed_encoder_count": 3,
         "isolated_interpreter": True, "model_api_calls": 0})
    print("Verified both released classifiers from the extracted archive using its bundled encoder weights.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("raw", "weights", "output"):
        p.add_argument("--" + name, required=True, type=lambda x: Path(x).resolve())
    args = p.parse_args()
    package(args.raw, args.weights, args.output)
    verify(args.raw, args.output)
