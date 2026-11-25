"""Read one task package or directory. A standalone table is the explicitly supplied split."""
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import zipfile
import re


TABLE_FORMATS = ('csv','tsv','parquet')


def evaluation_path(path,split):
    tokens=set(re.split(r'[^a-z0-9]+',str(path).casefold()))
    forbidden={'test','testing','prediction','holdout'}
    if split=='train':
        forbidden.update(('val','valid','validation','dev'))
    return bool(tokens & forbidden)


class LocalMembers:
    def __init__(self, path, split):
        self.root=path if path.is_dir() else path.parent
        self.standalone=None if path.is_dir() else (path.name,split+path.suffix.lower())
        if path.is_dir():
            files=(f for f in path.rglob('*') if f.is_file() and not f.is_symlink()
                   and f.resolve().is_relative_to(self.root))
            self.files={f.relative_to(path).as_posix():f for f in files}
        else:
            self.files={split+path.suffix.lower():path}
            for file in self.root.iterdir():
                if file.name.casefold() in ('readme.md','readme.txt') and file.is_file() and not file.is_symlink():
                    self.files[file.name]=file

    def namelist(self):
        return list(self.files)

    def getinfo(self,name):
        return SimpleNamespace(file_size=self.files[name].stat().st_size)

    def read(self,name):
        file=self.files[name].resolve()
        if not file.is_relative_to(self.root) or '..' in PurePosixPath(name).parts:
            raise ValueError('Input member is outside its task directory')
        return file.read_bytes()


@contextmanager
def open_package(value,split):
    path=Path(value).expanduser().resolve()
    if path.is_dir() or path.is_file() and path.suffix.lower().lstrip('.') in TABLE_FORMATS:
        yield LocalMembers(path,split)
    else:
        with zipfile.ZipFile(path) as archive:
            yield archive
