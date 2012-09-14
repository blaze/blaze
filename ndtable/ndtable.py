# An NDTable is either a LeafTable or a dictionary of NDTables (as values)
# LeafTables are either local-tables, expression-graphs, or remote tables
# Local tables are indexed-and-dtyped-bytes, indexed-NumPy arrays, or indexed-dynamic ndarrays.
# The essence of the NDTable is that operations are mapped to separate operations
# on the segments which when executed push the code to where the data lives.

"""
NDtables 
Attributes:
   * nd -- number of dimensions === number of independent indexing objects
   * children --- a mapping of names or sub-indexes to ndtable segments
   * name --- this is a unique URL for the ndtable or empty string if unassigned
   * indexes --- an nd-sequence of index-objects for each dimension.
   * flags --- attributes are true or false
"""
"""
Indexing:

functional:  getitem
full: hash
tree index:

subarrays:  [(ptr, strides)]
ascontiguous:  ptr, strides
"""

class Flags(object):
    def __init__(self, **kwds):
        for key, val in kwds.iteritems():
            kwds[key] = True if val else False
        self.__dict__.update(kwds)

    def __setattr__(self, attr, val):
        self.__dict__[attr] = True if val else False

############################
# Index Objects
############################
class Index(object):
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

class Auto(Index):
    pass

class Concrete(Index):
    values = []

class Mask(Index):
    mask = []

class Function(Index):
    func = None

class FuncIndex(Function):
    pass

class FuncCallBack(Function):
    pass

# wrapper around dyndarray and NumPy
class BasicArray(object):
    data = None
    shape = None
    strides = None
    dtype = Float()

class ITable(object):  # Table interface
    pass
        
class NDTable(ITable):
    nd = 1
    segments = {} # maps keys to NDTables
    name = ''
    dimexes = [Auto()]
    indexes = {}
    meta = {}
    # local, remote, expression
    flags = Flags()
    def __init__(self, **kwds):
        self.__dict__.update(kwds)


