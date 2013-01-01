import sqlite3

from blaze.desc.byteprovider import ByteProvider
from blaze.byteproto import CONTIGUOUS, CHUNKED, STREAM, ACCESS_READ
#from blaze.datadescriptor import SqlDataDescriptor

from blaze.layouts.categorical import Simple

class SqliteSource(ByteProvider):

    read_capabilities  = STREAM
    write_capabilities = STREAM
    access_capabilities = ACCESS_READ

    def __init__(self, data=None, dshape=None, params=None):
        #assert (data is not None) or (dshape is not None) or \
               #(params.get('storage'))

        if 'storage' in params and params.storage:
            self.conn = sqlite3.connect(params.storage)
        else:
            self.conn = sqlite3.connect(':memory:')

    def register_custom_types(self, name, ty, con, decon):
        sqlite3.register_adapter(ty, con)
        sqlite3.register_converter(name, decon)

    def read_desc(self, query):
        return SqlDataDescriptor('sqlite_dd', None, self.conn)

    def repr_data(self):
        return '<Deferred>'

    # Return the layout of the dataa
    def default_layout(self):
        return Simple()
        #return Simple(self.conn.read_schema())
