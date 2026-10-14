"""Short-lived NPY mappings and atomic numeric output files."""

from contextlib import contextmanager
from pathlib import Path

import numpy as np


@contextmanager
def mapped(path, mode='r'):
    array = np.load(path, mmap_mode=mode, allow_pickle=False)
    try:
        yield array
        if mode != 'r':
            array.flush()
    finally:
        array._mmap.close()


class DiskArray:
    """Array metadata plus chunk reads; never keep a dataset-sized map open."""

    def __init__(self, path, *, flat=False):
        self.path, self.flat = Path(path), flat
        with mapped(self.path) as array:
            self.original_shape = array.shape
            self.dtype = array.dtype
            self.size = array.size
        self.shape = (self.size,) if flat else self.original_shape
        self.ndim = len(self.shape)

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        with mapped(self.path) as array:
            source = array.reshape(-1) if self.flat else array
            return np.array(source[key], copy=True)

    def reshape(self, *shape):
        if shape != (-1,):
            raise ValueError('DiskArray supports only flattening; read a chunk for other reshapes.')
        return DiskArray(self.path, flat=True)


def create_npy(path, shape, dtype):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(path, mode='w+', shape=shape, dtype=dtype)
    array.flush()
    array._mmap.close()


def write_slice(path, key, values, *, flat=False):
    with mapped(path, 'r+') as array:
        target = array.reshape(-1) if flat else array
        target[key] = values


def atomic_npy(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.save(stream, values, allow_pickle=False)
        stream.flush()
    temporary.replace(path)
