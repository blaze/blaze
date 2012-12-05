import os
from tables import openFile, NoSuchNodeError
from blaze.datashape.record import from_pytables
from blaze.byteprovider import ByteProvider
from blaze.datashape import Fixed

class HDF5Source(ByteProvider):

    expectedrows = 10000
    compress = 'zlib'

    def __init__(self, shape, path, title=None):
        self.title = title
        self.path  = path
        self.shape = from_pytables(shape)
        self._shape = shape
        self.root, self.table = os.path.split(path)
        self.fd = None

        # TODO: lazy
        self.__alloc__()

    def get_or_create(self, node):
        try:
            return self.h5file.getNode('/' + self.table)
        except NoSuchNodeError:
            return self.h5file.createTable(
                '/',
                self.table,
                self._shape,
                title='',
            )

    def __alloc__(self):
        self.h5file = openFile(self.root, title=self.title, mode='a')
        self.fd = self.get_or_create(self.root)

    def __dealloc__(self):
        self.h5file.close()

    def __repr__(self):
        return "HDF5('%s')" % self.path
