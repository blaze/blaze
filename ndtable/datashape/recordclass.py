"""
Django style class declarations mapping onto Record datashape
types.

Also provides a mapping between HDF5 in terms of Blaze subset
that lets PyTables and Blaze interoperate.
"""

from utils import ReverseLookupDict
from coretypes import Record, Type, DataShape, Atom

from tables import IsDescription
from tables import description as pytables
import coretypes as blaze

pytable_rosetta = ReverseLookupDict({
    blaze.bool_      : pytables.BoolCol,
    blaze.string     : pytables.StringCol,

    blaze.int8       : pytables.Int8Col,
    blaze.int16      : pytables.Int16Col,
    blaze.int32      : pytables.Int32Col,
    blaze.int64      : pytables.Int64Col,

    blaze.uint8      : pytables.UInt8Col,
    blaze.uint16     : pytables.UInt16Col,
    blaze.uint32     : pytables.UInt32Col,
    blaze.uint64     : pytables.UInt64Col,

    blaze.float32    : pytables.Float32Col,
    blaze.float64    : pytables.Float64Col,
    #blaze.float128   : pytables.Float128Col,

    blaze.complex64  : pytables.Complex64Col,
    blaze.complex128 : pytables.Complex128Col,

    blaze.Enum       : pytables.EnumCol,
})

def from_pytables(description):
    fields = [
        pytable_rosetta[field] for field in description._f_walk()
    ]
    return type('Record', Decl, fields)

def to_pytables(record):
    fields = [
        pytable_rosetta[field] for field in record.fields
        if isinstance(field, Atom)
    ]

    substructs = [
        to_pytables(field) for field in record.fields
        if isinstance(field, DataShape)
    ]

    return type('Description', IsDescription, fields + substructs)

class DeclMeta(type):
    def __new__(meta, name, bases, namespace):
        abstract = namespace.pop('abstract', False)
        if abstract:
            cls = type(name, bases, namespace)
        else:
            cls = type.__new__(meta, name, bases, namespace)
            cls.construct.im_func(cls, namespace)
            if hasattr(cls, 'fields'):
                # Side-effectful operation to register the alias
                # with the parser
                rcd = Record(**cls.fields)
                Type.register(name, rcd)
        return cls

    def construct(cls, fields):
        pass

class Decl(object):
    __metaclass__ = DeclMeta

class RecordClass(Decl):
    """
    Record object, declared as class. Provied to the datashape
    parser through the metaclass.
    """
    fields = {}

    def construct(cls, fields):
        cls.fields = cls.fields.copy()
        for name, value in fields.items():
            if isinstance(value, DataShape):
                cls.add(name, value)

    @classmethod
    def add(cls, name, field):
        cls.fields[name] = field
