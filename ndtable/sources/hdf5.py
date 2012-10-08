import os
from tables import openFile
from ndtable.sources.canonical import Source
from ndtable.datashape.recordclass import from_pytables

class HDF5Source(Source):

    expectedrows = 10000
    compress = 'zlib'

    def __init__(self, shape, path, title=None):
        self.title = title
        self.path  = path
        self.shape = from_pytables(shape)
        self.root, self.table = os.path.split(path)
        self.fd = None

    def calculate(self, ntype):
        return self.shape

    def __alloc__(self):
        self.h5file = openFile(self.root, title=self.title, mode='r')
        self.fd = self.h5file.createTable(
            self.path,
            self.table,
            self.desc,
            title='',
        )

    def __dealloc__(self):
        self.h5file.close()

    def __repr__(self):
        return "HDF5('%s')" % self.path
