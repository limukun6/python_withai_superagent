"""Fetch and verify the public files used by the small cross-encoder experiment."""
import argparse
import hashlib
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile


MODEL = "cross-encoder/ms-marco-TinyBERT-L2-v2"
REVISION = "81d1926f67cb8eee2c2be17ca9f793c7c3bd20cc"
FILES = {"README.md":(3509,"eb670e4739d87fbe406f46c1566cc9ff6cfc7b5f"),
         "config.json":(787,"b4a8b8549bd075571ebedf155efcb571b7bf1463"),
         "tokenizer.json":(711396,"688882a79f44442ddc1f60d70334a7ff5df0fb47"),
         "model.safetensors":(17552980,"a0e7364ddf91ff7028f1102e1b91ac7a72e3db4061241bd84efe45c72c9af03a")}
SCALE_MODEL="thenlper/gte-base"
SCALE_REVISION="c078288308d8dee004ab72c6191778064285ec0c"
SCALE_FILES={"README.md":(68075,"cdd1740ec2e0bc0c9db50423219b562d3b170532"),
             "config.json":(618,"365ff9d060e6e6e052c90aa922d5c6dfad1b3f53"),
             "tokenizer.json":(711661,"c17ed520ed8438736732a54957a69306b8822215"),
             "model.safetensors":(218990904,"4a9e28bdb13c193da16ae4bdd6ba40ecceb5102b56bf8c1b039054290f644019")}


def verified(path, size, digest):
    if not path.is_file() or path.stat().st_size!=size:
        return False
    data=path.read_bytes()
    value=(hashlib.sha256(data).hexdigest() if len(digest)==64 else
           hashlib.sha1(f"blob {size}\0".encode()+data).hexdigest())
    return value==digest


def fetch(folder, endpoint, model=MODEL):
    revision,files=(REVISION,FILES) if model==MODEL else (SCALE_REVISION,SCALE_FILES)
    folder.mkdir(parents=True,exist_ok=True)
    for filename,(size,digest) in files.items():
        path=folder/filename
        if verified(path,size,digest):
            continue
        with NamedTemporaryFile(dir=folder,delete=False) as out:
            temp=Path(out.name)
        try:
            subprocess.run(["curl","--fail","--silent","--show-error","--location","--connect-timeout","12",
                            "--max-time","1200","--output",str(temp),f"{endpoint}/{model}/resolve/{revision}/{filename}"],
                           check=True)
            if not verified(temp,size,digest):
                raise ValueError("Public encoder checksum mismatch: "+filename)
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
        print(filename,size,flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder",type=Path,required=True)
    parser.add_argument("--model",choices=(MODEL,SCALE_MODEL),default=MODEL)
    parser.add_argument("--endpoint",choices=["https://huggingface.co","https://hf-mirror.com"],default="https://huggingface.co")
    args=parser.parse_args()
    fetch(args.folder,args.endpoint,args.model)
