"""Temporary idle-sleep request; never changes a persistent Windows power policy."""
from contextlib import contextmanager
import ctypes
import sys


ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


@contextmanager
def prevent_idle_sleep(set_state=None):
    """Keep this process's calling thread awake, then release in all exit paths."""
    if set_state is None:
        if sys.platform != 'win32':
            raise OSError('The unattended sleep guard currently supports Windows only')
        set_state = ctypes.windll.kernel32.SetThreadExecutionState
        set_state.argtypes = [ctypes.c_uint]
        set_state.restype = ctypes.c_uint
    if not set_state(ES_CONTINUOUS | ES_SYSTEM_REQUIRED):
        raise OSError('Windows refused the temporary idle-sleep request; job not started')
    try:
        yield
    finally:
        if not set_state(ES_CONTINUOUS):
            raise OSError('Windows failed to release this thread idle-sleep request')
