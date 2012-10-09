"""
Django style class declarations mapping onto Record datashape
types.

Also provides a mapping between HDF5 in terms of Blaze subset
that lets PyTables and Blaze interoperate.
"""

from tables import IsDescription

from utils import ReverseLookupDict
from coretypes import Record, Type, DataShape, Atom
from ndtable import rosetta

def pytables_deconstruct(col):
    name, atom = col
    try:
        stem = rosetta.pytables[type(atom)]
    except KeyError:
        raise Exception('Could not cast')
    return name, stem

def from_pytables(description):
    fields = dict(
        pytables_deconstruct(field) for field in
        description.columns.iteritems()
    )
    return Record(**fields)

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
