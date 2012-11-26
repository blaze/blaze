"""
Django style class declarations mapping onto Record datashape
types.

Also provides a mapping between HDF5 in terms of Blaze subset
that lets PyTables and Blaze interoperate.
"""

from numpy import dtype
from coretypes import Record, Type, DataShape, Atom, to_numpy

class DeclMeta(type):
    def __new__(meta, name, bases, namespace):
        abstract = namespace.pop('__abstract', False)
        # for tests
        dummy = namespace.pop('__dummy', False)

        if abstract or dummy:
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

class RecordDecl(object):
    """
    Record object, declared as class. Provied to the datashape
    parser through the metaclass.
    """
    fields = {}
    __metaclass__ = DeclMeta

    def construct(cls, fields):
        cls.fields = cls.fields.copy()
        for name, value in fields.items():
            if isinstance(value, DataShape):
                cls.add(name, value)

    @classmethod
    def add(cls, name, field):
        cls.fields[name] = field

    @classmethod
    def to_numpy(self):
        """
        Convert a record class definition into a structured array
        dtype
        """
        n = dtype([
            (label, ty.to_dtype()) for label, ty in
            self.fields.iteritems()
        ])
        return n
