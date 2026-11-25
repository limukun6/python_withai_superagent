"""Read one task package or directory. A standalone table is the explicitly supplied split."""
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
import zipfile
import re


TABLE_FORMATS = ('csv','tsv','parquet')


def evaluation_path(path,split):
    tokens=set(re.split(r'[^a-z0-9]+',str(path).casefold()))
    forbidden={'test','testing','prediction','predictions','predict','inference','holdout'}
    if split=='train':
        forbidden.update(('val','valid','validation','dev'))
    return bool(tokens & forbidden)


class TrainingMembers:
    def __init__(self, archive):
        self.archive = archive
        self.standalone = getattr(archive, 'standalone', None)
        self.names = [n for n in archive.namelist() if not evaluation_path(n, 'train')]
        self.allowed = set(self.names)

    def namelist(self):
        return list(self.names)

    def read(self, name):
        if name not in self.allowed:
            raise ValueError('Member is outside the training split')
        try:
            return self.archive.read(name)
        except RuntimeError as error:
            if 'encrypt' in str(error).lower() or 'password' in str(error).lower():
                raise ValueError('An encrypted training member requires its decryption key') from error
            raise

    def getinfo(self, name):
        if name not in self.allowed:
            raise ValueError('Metadata is outside the training split')
        return self.archive.getinfo(name)


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
def open_package(value,split,*,shared=False,contract=None):
    from contextlib import nullcontext
    from .shared_table import stage
    path=Path(value).expanduser().resolve()
    if split == 'train' and path.is_file() and path.suffix.lower().lstrip('.') in TABLE_FORMATS and evaluation_path(path.name, 'train'):
        raise ValueError('A standalone evaluation table cannot be a training source')
    if path.is_dir() or path.is_file() and path.suffix.lower().lstrip('.') in TABLE_FORMATS:
        source=nullcontext(LocalMembers(path,split))
    else:
        from .format_sources import package_files
        with path.open('rb') as stream:
            prefix = stream.read(512)
        tar_or_gzip = prefix[257:262] == b'ustar' or prefix.startswith(bytes([31,139]))
        source=package_files(path) if tar_or_gzip else zipfile.ZipFile(path)
    with source as opened:
        archive = TrainingMembers(opened) if split == 'train' else opened
        with (stage(archive,split,contract) if shared or contract else nullcontext(archive)) as selected:
            yield selected
