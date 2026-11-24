"""Public, pinned CPU encoder assets used by the optional preference portfolio."""
from functools import lru_cache
import hashlib
from pathlib import Path

from superagent.ab_sentence import ALTERNATIVES


BGE = "bge-small-en-v1.5"
MPNET = "all-mpnet-base-v2"
ROBERTA = "roberta-base"
MODELS = {
    BGE: {"repo": ALTERNATIVES["bge"]["model"], "revision": ALTERNATIVES["bge"]["revision"],
          "files": ALTERNATIVES["bge"]["files"]},
    MPNET: {"repo": "sentence-transformers/all-mpnet-base-v2",
            "revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
            "files": {"config.json": (571, "b9fd4298819da011007a6a4ceb728c860914fc88"),
                      "tokenizer.json": (466021, "7c8e194053bc80e27e19bb2125469e4f289ab2b3"),
                      "model.safetensors": (437971872, "78c0197b6159d92658e319bc1d72e4c73a9a03dd03815e70e555c5ef05615658")}},
    ROBERTA: {"repo": "FacebookAI/roberta-base", "revision": "e2da8e2f811d1448a5b465c236feacd80ffbac7b",
              "files": {"config.json": (481, "8db5e7ac5bfc9ec8b613b776009300fe3685d957"),
                        "tokenizer.json": (1355863, "ad0bcbeb288f0d1373d88e0762e66357f55b8311"),
                        "model.safetensors": (498818054, "5bde1d28afb363d0103324efeb5afc8b2b397fe5e04beabb9b1ef355255ade81")}},
}


@lru_cache(maxsize=32)
def file_identity(path, size, inode, modified, changed, digest_length):
    if digest_length == 64:
        digest = hashlib.sha256()
    else:
        digest = hashlib.sha1()
        digest.update(f"blob {size}\0".encode())
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def matches(path, expected):
    if not path.is_file():
        return False
    stat = path.stat()
    size, digest = expected
    return stat.st_size == size and file_identity(path, size, stat.st_ino, stat.st_mtime_ns,
                                                   stat.st_ctime_ns, len(digest)) == digest


def verified(weights, name):
    folder = Path(weights) / name
    for filename, identity in MODELS[name]["files"].items():
        if not matches(folder/filename, identity):
            raise ValueError(f"Missing or mismatched {name}/{filename}")
    return folder
