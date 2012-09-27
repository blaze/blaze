class NDTable(object):
    def __init__(self, byte_interfaces, datashape, index=None, metadata=None):
        self.datashape = datashape
        self.byte_interfaces = byte_interfaces
        self.metadata = metadata
        self.index = index

    def __getitem__(self, index):
        pass

    def __getslice__(self, i, j):
        pass
