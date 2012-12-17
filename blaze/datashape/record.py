"""
Django style class declarations mapping onto Record datashape types.
"""

from numpy import dtype
from coretypes import Record, Type, DataShape
from parse import parse

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
                rcd = Record(**dict(cls.fields, **cls._derived))
                Type.register(name, rcd)
        return cls

class RecordDecl(object):
    """
    Record object, declared as class. Provied to the datashape parser
    through the metaclass.
    """
    __metaclass__ = DeclMeta

    def construct(cls, fields):
        cls.fields = {}
        cls._derived = {}
        for name, value in fields.items():
            if isinstance(value, DataShape):
                cls.add(name, value)

    @classmethod
    def add(cls, name, field):
        cls.fields[name] = field

    @classmethod
    def to_dtype(self):
        """
        Convert a record class definition into a structured array dtype
        """
        n = dtype([
            (label, ty.to_dtype()) for label, ty in
            self.fields.iteritems()
        ])
        return n

def derived(sig=None):
    def wrapper(fn):
        if sig is not None:
            return parse(sig)
        else:
            raise NotImplementedError
    return wrapper
