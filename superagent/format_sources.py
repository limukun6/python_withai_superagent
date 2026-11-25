"""Select bounded task members by split before inspecting their contents."""
from contextlib import contextmanager
import gzip
import io
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import stat
import tarfile
from types import SimpleNamespace
import zipfile


MAX_BYTES = 96 * 1024 * 1024
MAX_MEMBERS = 4096
ROLE_WORDS = {
    'train': {'train', 'training', '训练', '训练集'},
    'test': {'test', 'testing', 'predict', 'prediction', 'predictions', 'inference', '测试', '测试集', '预测'},
    'validation': {'val', 'valid', 'validation', 'dev', 'holdout', '验证', '验证集'},
}


class UnsupportedFormat(ValueError):
    """The adaptive reader has no authorized interpretation for this package."""


class NativeContract(ValueError):
    """An existing explicit or specialized contract must resolve its own errors."""


def roles(name):
    tokens = set(re.split(r'[^\w]+|_', name.casefold()))
    return {key for key, words in ROLE_WORDS.items() if words & tokens}


def checked_name(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or PureWindowsPath(name).drive or '..' in path.parts
            or '\\' in name or '\x00' in name):
        raise ValueError('Unsafe input member path')
    return name


def checked_list(entries):
    names = []
    for name in entries:
        if len(names) >= MAX_MEMBERS:
            raise ValueError('Too many package members')
        checked_name(name)
        if name in names:
            raise ValueError('Duplicate member name')
        names.append(name)
    return names


def checked_bytes(stream, limit=MAX_BYTES):
    raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('Format inspection exceeds its byte limit')
    return raw


def uncompress(raw):
    if raw.startswith(b'\x1f\x8b'):
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as contents:
            return checked_bytes(contents)
    return raw


class Package:
    def __init__(self, names, reader, size, standalone=None):
        self.names = checked_list(names)
        self.reader = reader
        self.size = size
        self.standalone = standalone

    def read(self, name, limit=MAX_BYTES):
        if name not in self.names:
            raise ValueError('Requested member is outside the package')
        return self.reader(name, limit)

    def namelist(self):
        return list(self.names)

    def getinfo(self, name):
        if name not in self.names:
            raise ValueError('Requested metadata is outside the package')
        return SimpleNamespace(file_size=self.size(name))

    def documentation(self):
        docs = [n for n in self.names if PurePosixPath(n).name.casefold() in
                {'readme.md', 'readme.txt', 'task.md', 'instructions.md'} and not roles(n)]
        if len(docs) > 1:
            raise ValueError('Multiple task instructions need an explicit manifest')
        if not docs:
            return ''
        return decode_text(self.read(docs[0], 256 * 1024)).replace(chr(96), '').replace('*', '')

    def data_names(self):
        return [n for n in self.names if not any(p.startswith('.') or p == '__MACOSX' for p in PurePosixPath(n).parts)
                and PurePosixPath(n).name.casefold() not in
                {'readme.md', 'readme.txt', 'task.md', 'instructions.md', 'info.json', 'manifest.json'}]


def decode_text(raw):
    encoding = 'utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig'
    text = raw.decode(encoding)
    if '\x00' in text:
        raise UnsupportedFormat('No supported text encoding or binary signature')
    return text


def zip_names(archive):
    items = archive.infolist()
    if len(items) > MAX_MEMBERS:
        raise ValueError('Too many ZIP entries')
    for item in items:
        checked_name(item.filename)
        if stat.S_ISLNK(item.external_attr >> 16):
            raise ValueError('Package links are unsupported')
    return checked_list(i.filename for i in items if not i.is_dir())


def zip_read(archive, name, limit):
    entry = archive.getinfo(name)
    if entry.file_size > limit or entry.flag_bits & 1:
        raise UnsupportedFormat('Member is encrypted or exceeds the byte limit')
    with archive.open(entry) as stream:
        return checked_bytes(stream, limit)


@contextmanager
def package_files(value):
    path = Path(value).expanduser().resolve(strict=True)
    if path.is_dir():
        paths = (p for p in path.rglob('*') if p.is_file() and not p.is_symlink()
                 and p.resolve().is_relative_to(path))
        names = checked_list(p.relative_to(path).as_posix() for p in paths)
        def read(name, limit):
            member = (path / name).resolve(strict=True)
            if not member.is_relative_to(path):
                raise ValueError('Task member escaped its directory')
            with member.open('rb') as stream:
                return checked_bytes(stream, limit)
        yield Package(names, read, lambda n:(path / n).stat().st_size)
        return
    with path.open('rb') as stream:
        prefix = stream.read(512)
    body = None
    if prefix.startswith(b'\x1f\x8b'):
        with gzip.open(path, 'rb') as stream:
            body = checked_bytes(stream)
        prefix = body[:512]
    if prefix.startswith(b'PK') and zipfile.is_zipfile(io.BytesIO(body) if body is not None else path):
        with zipfile.ZipFile(io.BytesIO(body) if body is not None else path) as archive:
            members = zip_names(archive)
            if 'xl/workbook.xml' not in members:
                yield Package(members, lambda name, limit: zip_read(archive, name, limit),
                              lambda n:archive.getinfo(n).file_size)
                return
    if prefix[257:262] == b'ustar' or path.name.casefold().endswith(('.tar', '.tar.gz', '.tgz')):
        with tarfile.open(name=None if body is not None else path,
                          fileobj=io.BytesIO(body) if body is not None else None, mode='r:') as archive:
            members = {}
            for index, entry in enumerate(archive):
                if index >= MAX_MEMBERS or not (entry.isdir() or entry.isreg()):
                    raise UnsupportedFormat('Too many TAR members or unsupported link/device')
                name = checked_name(entry.name)
                if entry.isreg():
                    if name in members:
                        raise ValueError('Duplicate TAR member')
                    members[name] = entry
            def read_tar(name, limit):
                if members[name].size > limit:
                    raise ValueError('TAR member exceeds its byte limit')
                with archive.extractfile(members[name]) as stream:
                    return checked_bytes(stream, limit)
            yield Package(members, read_tar, lambda n:members[n].size)
            return
    docs = [p.name for p in path.parent.iterdir() if p.name.casefold() in
            {'readme.md', 'readme.txt', 'instructions.md', 'task.md'} and p.is_file() and not p.is_symlink()]
    def read_standalone(name, limit):
        if name == path.name and body is not None:
            if len(body) > limit:
                raise ValueError('Member exceeds its byte limit')
            return body
        with (path.parent / name).open('rb') as stream:
            return checked_bytes(stream, limit)
    yield Package([path.name, *(n for n in docs if n != path.name)], read_standalone,
                  lambda n:len(body) if n == path.name and body is not None else (path.parent / n).stat().st_size,
                  (path.name, path.name))


def select(names, role, *, explicit=None, single=False):
    """A declaration or an unambiguous split marker takes precedence over an isolated input."""
    if explicit is not None:
        checked_name(explicit)
        match = [n for n in names if n == explicit or n.endswith('/' + explicit)]
        if len(match) != 1 or roles(match[0]) - {role}:
            raise ValueError('Declared source is absent, ambiguous, or belongs to another split')
        return match[0]
    named = [n for n in names if roles(n) == {role}]
    if len(named) == 1:
        return named[0]
    if not named and single and len(names) == 1 and not roles(names[0]):
        return names[0]
    raise UnsupportedFormat('No unambiguous ' + role + ' source')
