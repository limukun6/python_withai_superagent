"""Read legacy NPY object arrays of ordinary categorical values without importing pickle globals."""
import io
import math
import pickle
import pickletools

import numpy as np
from numpy.lib import format


_ARRAY_TOKEN = object()


class _ObjectDtype:
    def __init__(self, name, *flags):
        if name not in ('O','O4','O8','|O','|O4','|O8'):
            raise ValueError('Expected the legacy object dtype')

    def __setstate__(self, state):
        if not isinstance(state, tuple) or len(state) != 8 or state[1:5] != ('|',None,None,None):
            raise ValueError('Unsupported legacy dtype state')


class _ArrayValues:
    def __setstate__(self, state):
        if not isinstance(state, tuple) or len(state) != 5:
            raise ValueError('Unsupported legacy array state')
        version, shape, dtype, order, values = state
        if (version != 1 or type(dtype) is not _ObjectDtype or type(order) is not bool
                or type(shape) is not tuple or not 1 <= len(shape) <= 4
                or any(type(n) is not int or n < 0 for n in shape)
                or type(values) is not list or math.prod(shape) != len(values)
                or any(type(v) not in (str,bytes,int,float,bool,type(None)) for v in values)):
            raise ValueError('Legacy categorical array requires plain scalar cells')
        self.shape, self.fortran, self.values = shape, order, values


def _reconstruct(tag, empty, raw_type):
    if tag is not _ARRAY_TOKEN or empty != (0,) or raw_type != b'b':
        raise ValueError('Unsupported legacy array constructor')
    return _ArrayValues()


class _ScalarArrayReader(pickle.Unpickler):
    def find_class(self, module, name):
        if module in ('numpy.core.multiarray','numpy._core.multiarray') and name == '_reconstruct':
            return _reconstruct
        if module == 'numpy' and name == 'ndarray':
            return _ARRAY_TOKEN
        if module == 'numpy' and name == 'dtype':
            return _ObjectDtype
        raise ValueError('Unsupported class in legacy categorical array')


def load_array(raw):
    buffer = io.BytesIO(raw)
    version = format.read_magic(buffer)
    if version not in ((1,0),(2,0),(3,0)):
        raise ValueError('Unsupported NPY version')
    read_header = format.read_array_header_1_0 if version == (1,0) else format.read_array_header_2_0
    shape, order, dtype = read_header(buffer)
    if not dtype.hasobject:
        buffer.seek(0)
        return np.load(buffer, allow_pickle=False)
    if dtype != np.dtype(object) or len(raw) > 96 * 1024 * 1024 or math.prod(shape) > 12_000_000:
        raise ValueError('Unsupported legacy object-array size or dtype')
    body = buffer.read()
    try:
        for op, _, _ in pickletools.genops(body):
            if op.name in {'EXT1','EXT2','EXT4','PERSID','BINPERSID','NEXT_BUFFER','READONLY_BUFFER'}:
                raise ValueError('Unsupported legacy array encoding')
        array = _ScalarArrayReader(io.BytesIO(body)).load()
        if type(array) is not _ArrayValues or array.shape != shape or array.fortran != order:
            raise ValueError('Legacy array metadata differs from its NPY header')
        result = np.asarray(array.values, dtype=object).reshape(shape)
        return np.asfortranarray(result) if order else result
    except (pickle.UnpicklingError, AttributeError, TypeError, IndexError, OverflowError) as error:
        raise ValueError('Invalid legacy categorical array') from error
