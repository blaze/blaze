class NDTable(object):
    def __init__(self, byte_interface, datashape):
        self.datashape = datashape
        self.byte_interface = byte_interface

    def __getitem__(self, index):
        pass

    def __getslice__(self, i, j):
        pass
